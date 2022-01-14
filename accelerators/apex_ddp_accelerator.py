# -*- coding: utf-8 -*-
# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import random
import sys
from typing import Tuple, Union, Optional, Any
import numpy as np

import torch
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

Net = torch.nn.Module

from .accelerator import Accelerator

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
    from apex.parallel import convert_syncbn_model
except ImportError:
    print('no apex! Please install from https://www.github.com/nvidia/apex')


class ApexDDPAccelerator(Accelerator):
    """
    ApexDDPAccelerator, use apex DistributedDataParallel
    """

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.accelerator_rng_seed = self.cfg.RNG_SEED
        self.accelerator_syncbn = self.cfg.SYNCBN
        self.accelerator_fp16_opt_level = self.cfg.FP16_OPT_LEVEL
        self.accelerator_fp16_loss_scale = self.cfg.FP16_LOSS_SCALE

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR,
               local_rank: int, world_size: int, rank: int) -> Tuple[Apex_DDP, Optimizer, LambdaLR]:
        """
        set up ApexDDPAccelerator, including process_group and apex_ddp
        """
        torch.backends.cudnn.benchmark = False
        random.seed(self.accelerator_rng_seed)
        np.random.seed(self.accelerator_rng_seed)
        torch.random.manual_seed(self.accelerator_rng_seed)
        torch.cuda.manual_seed_all(self.accelerator_rng_seed)
        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))

        torch.cuda.set_device(local_rank)
        model = model.cuda()
        if not torch.distributed.is_initialized():
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
            print(
                f'ApexDDPAccelerator distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
            sys.stdout.flush()

        self.broadcast(model)
        apex_model, optimizer = self.configure_ddp(model, optimizer)

        if self.accelerator_syncbn:
            apex_model = self.configure_sync_batchnorm(apex_model)
        return apex_model, optimizer, lr_scheduler

    def broadcast(self, model: Net, src=0) -> None:
        for v in model.state_dict().values():
            distributed.broadcast(v, src)

    def configure_ddp(self, model: Net, optimizer: Optimizer) -> Tuple[Apex_DDP, Optimizer]:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=self.accelerator_fp16_opt_level,
                                          keep_batchnorm_fp32=None,  # from True to None
                                          loss_scale=self.accelerator_fp16_loss_scale,
                                          max_loss_scale=1024.0,
                                          min_loss_scale=1.0)

        apex_model = Apex_DDP(model, delay_allreduce=True)
        self.ddp_model = apex_model
        return apex_model, optimizer

    def configure_sync_batchnorm(self, model: Net) -> Net:
        model = convert_syncbn_model(model)
        return model

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net, grad_norm: float) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                    grad_norm)
        return float(total_norm)
