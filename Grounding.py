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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import load_pretrained
from models.model_retrieval import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.utils import collect_tensor_result, grounding_eval
from scheduler import create_scheduler
from optim import create_optimizer

from refTools.refer_python3 import REFER

from pdb import set_trace as breakpoint

from utils.hdfs_io import hmkdir, hcopy, hexists


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    for i,(image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)

        loss_itc, loss_itm = model(image, text_input.input_ids, text_input.attention_mask, idx=idx)
        loss = loss_itc + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def val(model, data_loader, tokenizer, device, gradcam_mode, block_num, num_patches):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    
    if gradcam_mode == 'itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
     
    result = []
    for image, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        
        if gradcam_mode == 'itm':
            image_embeds, image_atts = model.get_vision_embeds(image)
            vl_embeddings = model.get_cross_embeds(image_embeds, image_atts, text_ids=text_input.input_ids,
                                            text_atts=text_input.attention_mask)[:,0,:]
            vl_output = model.itm_head(vl_embeddings)
            loss = vl_output[:, 1].sum()
            
            model.zero_grad()
            loss.backward()    

            with torch.no_grad():
                mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)

                grads = model.text_encoder.base_model.base_model.encoder.layer[
                    block_num].crossattention.self.get_attn_gradients().detach()
                cams = model.text_encoder.base_model.base_model.encoder.layer[
                    block_num].crossattention.self.get_attention_map().detach()

                cams = cams[:, :, :, 1:].reshape(image.size(0), model.num_attention_heads, -1, num_patches,
                                                 num_patches) * mask
                grads = grads[:, :, :, 1:].clamp(min=0).reshape(image.size(0), model.num_attention_heads, -1,
                                                                num_patches, num_patches) * mask

                gradcam = cams * grads
                gradcam = gradcam.mean(1).mean(1)

        elif gradcam_mode == 'itc':
            raise NotImplementedError

        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id': r_id.item(), 'pred': cam})
  
    if gradcam_mode == 'itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False

    return result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if world_size > 8:
        assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.block_num > 0:
        config['block_num'] = args.block_num

    if args.bs > 0:
        config['batch_size'] = args.bs // world_size

    if args.epochs > 0:
        config['schedular']['epochs'] = args.epochs

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    grd_train_dataset, grd_test_dataset = create_dataset('grounding', config) 
    datasets = [grd_train_dataset, grd_test_dataset]

    train_dataset_size = len(grd_train_dataset)
    train_batch_size = config['batch_size']

    if utils.is_main_process():
        print(f"### data {train_dataset_size}, batch size, {train_batch_size} x {world_size}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size'], config['batch_size']],
                                              num_workers=[4, 4], is_trains=[True, False], collate_fns=[None, None])

    # refcoco evaluation tools
    refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
    dets = json.load(open(config['det_file'], 'r'))
    cocos = json.load(open(config['coco_file'], 'r'))

    print("Creating model")
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)
    print("### output_hdfs, ", args.output_hdfs, flush=True)
    start_time = time.time()

    if args.evaluate:
        print("Start evaluating")
        print("### block_num, ", config['block_num'])

        num_patches = config['image_res'] // config['patch_size']
        result = val(model_without_ddp, test_loader, tokenizer, device, args.gradcam_mode, config['block_num'],
                     num_patches=num_patches)
        results = collect_tensor_result(result, 'grounding_eval', local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs, write_to_hdfs=world_size > 8)

        if utils.is_main_process():
            grounding_acc = grounding_eval(results, dets, cocos, refer, alpha=0.5, mask_size=num_patches)
            log_stats = {**{f'{k}': v for k, v in grounding_acc.items()}}
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()

    else:
        print("Start training")
        print("### block_num, ", config['block_num'])
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (train_batch_size * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            num_patches = config['image_res'] // config['patch_size']
            result = val(model_without_ddp, test_loader, tokenizer, device, args.gradcam_mode, config['block_num'], num_patches=num_patches)
            results = collect_tensor_result(result, 'epoch%d' % epoch, local_wdir=args.result_dir, hdfs_wdir=args.output_hdfs, write_to_hdfs=world_size > 8)

            if utils.is_main_process():
                grounding_acc = grounding_eval(results, dets, cocos, refer, alpha=0.5, mask_size=num_patches)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'{k}': v for k, v in grounding_acc.items()},
                             'epoch': epoch}

                if grounding_acc['val_d'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = grounding_acc['val_d']
                    best_epoch = epoch

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/Grounding.yaml')
    parser.add_argument('--output_dir', default='output/refcoco')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--gradcam_mode', default='itm', choices=['itm'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--block_num', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--epochs', default=-1, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    if len(args.output_hdfs):
        hmkdir(args.output_hdfs)

    main(args, config)