import argparse
import os
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import load_pretrained
from models.model_vqa import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hexists

from dataset.utils import collect_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        loss = model(image, question_input, answer_input, train=True, k=n, weights=weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []

    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    print("Creating vqa datasets")
    train_dataset, vqa_test_dataset = create_dataset('vqa', config)
    datasets = [train_dataset, vqa_test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    print("Creating model")

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("### pad_token_id, ", train_dataset.pad_token_id)
    print("### eos_token, ", train_dataset.eos_token)
    config['pad_token_id'] = train_dataset.pad_token_id
    config['eos'] = train_dataset.eos_token
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    if args.evaluate:
        print("Start evaluating")
        vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        result = collect_result(vqa_result, 'vqa_eval', local_wdir=args.result_dir,
                                hdfs_wdir=args.output_hdfs,
                                write_to_hdfs=world_size > 8, save_result=True)

        dist.barrier()

    else:
        print("Start training")
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        for epoch in range(start_epoch, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    # 'epoch': epoch,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch,
                                             training_states=optimizer.state_dict())

            if epoch >= config['start_eval']:
                vqa_result = evaluation(model, test_loader, tokenizer, device, config)
                result = collect_result(vqa_result, 'vqa_result_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                         write_to_hdfs=world_size > 8, save_result=True)

            dist.barrier()

        os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    hmkdir(args.output_dir)
    hmkdir(args.result_dir)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)