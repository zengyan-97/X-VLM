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
from models.model_captioning import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from utils import ScstRewardCriterion
from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hexists

from dataset.utils import collect_result, coco_caption_eval
from dataset import create_dataset, create_sampler, create_loader


from scheduler import create_scheduler
from optim import create_optimizer


def scst_train_iter(image, captions_gt, model, scst_criterion, config):
    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    if config['sc_baseline_type'] == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res = model_without_ddp.generate(image, sample=False, num_beams=1,
                                       max_length=config['max_length'],
                                       min_length=config['min_length'], greedy=True)

    else:
        greedy_res = None

    model.train()
    sample_res, sample_logprobs = model_without_ddp.generate(image, sample=True, num_beams=1, num_return_sequences=config['sc_train_sample_n'],
                                            max_length=config['max_length'], min_length=config['min_length'])

    assert sample_logprobs.requires_grad == True

    loss = scst_criterion(captions_gt, greedy_res, sample_res, sample_logprobs)
    return loss


def train(model, data_loader, optimizer, epoch, device, scheduler, scst_criterion, config, global_step):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, captions_gt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)

        loss = scst_train_iter(image, captions_gt, model, scst_criterion, config)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step += 1
        if (global_step % config['eval_steps'] == 0) or (global_step >= config['num_training_steps']):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return global_step, {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50
    
    result = []

    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device, non_blocking=True)

        captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

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
    
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco_scst', config)
    datasets = [train_dataset, val_dataset, test_dataset]

    train_dataset_size = len(train_dataset)
    world_size = utils.get_world_size()

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4, 4, 4], is_trains=[True, False, False],
                                              collate_fns=[train_dataset.collate_fn, None, None])

    print("Creating model")
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate, load_capt_pretrain=args.load_capt_pretrain)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)

    if args.evaluate:
        print("Start evaluating")
        test_result = evaluation(model, test_loader, device, config)
        test_result_file = collect_result(test_result, 'test_eval', local_wdir=args.result_dir,
                                          hdfs_wdir=args.output_hdfs,
                                          write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

        if utils.is_main_process():
            coco_test = coco_caption_eval(config['test_gt_file'], test_result_file)
            log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
            print(log_stats, flush=True)

        dist.barrier()

    else:
        print("Start SCST training", flush=True)
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=config['cider_cached_tokens'],
            baseline_type=config['sc_baseline_type'])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        config['num_training_steps'] = arg_sche.num_training_steps
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        checkpointer = Checkpointer(args.output_hdfs if hexists(args.output_hdfs) else args.output_dir)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        global_step = 0
        step_per_epoch = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        while global_step < config['num_training_steps']:
            epoch = global_step // step_per_epoch
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            global_step, train_stats = train(model, train_loader, optimizer, epoch, device, lr_scheduler, scst_criterion, config, global_step=global_step)

            print(f"### epoch: {epoch}, global_step: {global_step}", flush=True)

            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'config': config}
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            test_result = evaluation(model, test_loader, device, config)
            test_result_file = collect_result(test_result, 'test_epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs,
                                     write_to_hdfs=world_size > 8, save_result=True, remove_duplicate='image_id')

            if utils.is_main_process():
                coco_test = coco_caption_eval(config['test_gt_file'], test_result_file)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             # **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             'step': global_step}

            dist.barrier()

            if utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

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

    parser.add_argument('--load_capt_pretrain', action='store_true')
    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')

    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    hmkdir(args.output_dir)
    hmkdir(args.result_dir)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)