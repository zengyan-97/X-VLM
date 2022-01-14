import argparse
import copy
import os
import sys

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer

from models import load_pretrained
from models.model_bbox_pretrain import XVLM

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy, hexists
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        # from transformers import get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def train(model, dataloader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    # train
    model.train()  
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train step: [{}]'.format(start_epoch)
    print_freq = 50   

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = 0 # start from 0

    for i, region_batch in enumerate(metric_logger.log_every(dataloader, print_freq, header, step_per_epoch, epoch_info)):

        image, region_batch = region_batch[0].to(device, non_blocking=True), [
            t.to(device) if t is not None else None for t in region_batch[1:]]
        idx_to_group_img, text_ids, text_atts, _, _, _, _, target_bbox, is_image = region_batch

        if config['calc_image_bbox_loss']:
            is_image = None

        optimizer.zero_grad()
        loss_bbox, loss_giou = model(image, text_ids=text_ids, text_atts=text_atts,
                                        idx_to_group_img=idx_to_group_img, target_bbox=target_bbox, is_image=is_image)

        loss = loss_bbox + loss_giou
        accelerator.backward_step(loss, optimizer)

        accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
        if accelerator_clip_grad_norm > 0:
            accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if utils.is_main_process():
            if (global_step+1) % step_per_epoch == 0:
                current_epoch = global_step // step_per_epoch
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        global_step += 1

    if utils.is_main_process():
        model_without_ddp = model
        if hasattr(model, 'module'):
            model_without_ddp = model.module

        save_obj = {
            'model': model_without_ddp.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': scheduler.state_dict(),
            'config': config,
            # 'epoch': current_epoch,
        }
        checkpointer.save_checkpoint(model_state=save_obj,
                                     epoch='latest',
                                     training_states=optimizer.state_dict())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['train_file_regions'] = ','.join(config['train_file_regions'])
    config['batch_size'] = config['regions']['batch_size']

    if utils.is_main_process():
        print(f"### train_file_regions: {config['train_file_regions']}")
        sys.stdout.flush()

    world_size = utils.get_world_size()
    if world_size > 8:
        # you can comment out this assertion if you run the two scripts manually
        assert args.output_dir.startswith('hdfs') and hexists(args.output_dir), \
            "to read ckpt for each node when running Grounding_bbox.py subsequently"

    # fix the seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating dataset")
    dataset = create_dataset('grounding_bbox_pretrain', config)

    loader = torch.utils.data.DataLoader(dataset, batch_size=config['regions']['max_images'],
                                               num_workers=config['regions']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=dataset.collate_fn)

    print("Creating model")
    model = XVLM(config=config)
    # print(model)
    model.load_pretrained(args.checkpoint, config)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)
    reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    print("Start training")
    start_time = time.time()

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    epoch_info = (start_epoch, max_epoch)

    train(model, loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/bbox_pretrain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    main(args, config)