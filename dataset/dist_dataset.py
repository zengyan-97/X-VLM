#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import sys
from typing import List, Any
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset

from utils.hdfs_io import hopen, hlist_files


class DistLineReadingDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """
    def __init__(self,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = hlist_files(data_path.split(','))
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
        self.is_hdfs = data_path.startswith('hdfs')

        self.repeat = repeat
        print('[DATA]--all dataset containing {} files.'.format(len(self.files)))
        if len(self.files) % self.world_size != 0:
            print('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        sys.stdout.flush()

    def generate(self):
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)

        while True:
            if self.shuffle:
                random.shuffle(cur_dataloader_files)
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0:
                    print('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                cur_worker_files = split_shard(
                    cur_dataloader_files, worker_info.id, worker_info.num_workers)
                if worker_info.id == 0:
                    print("[DataLoader] --> Rank:{}  Workers:[{} ~ {}][{}]  Size of process file:{}  ...".format(
                        self.rank, 0, worker_info.num_workers - 1, worker_info.id, len(cur_dataloader_files)))
            else:
                cur_worker_files = cur_dataloader_files

            if self.shuffle:
                random.shuffle(cur_worker_files)
            for filepath in cur_worker_files:
                if self.is_hdfs:
                    with hopen(filepath, 'r') as reader:
                        for line in reader:
                            yield line.decode()
                    continue
                with open(filepath, 'r') as reader:
                    for line in reader:
                        yield line

            if not self.repeat:
                break

    def __iter__(self):
        return self.generate()  


def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]
