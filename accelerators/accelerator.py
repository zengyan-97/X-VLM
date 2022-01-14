# -*- coding: utf-8 -*-
# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

from logging import Logger

import torch
from torch.optim import Optimizer

Net = torch.nn.Module


class Accelerator:
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
    
    def set_up(self, model: Net):
        raise NotImplementedError("Set Up method not implement in Accelerator, please check! ")

    def broadcast(self):
        raise NotImplementedError("Broadcast method not implement in Accelerator, please check! ")

    def backward_step(self, loss: torch.Tensor):
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net, grad_norm: float) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        grad_norm)
        return float(total_norm)
