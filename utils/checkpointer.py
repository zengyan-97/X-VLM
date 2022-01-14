# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

from typing import Union, Dict, List, Tuple, Any, Callable
import logging
import os
import re
import time

import torch

from utils.hdfs_io import hexists, hmkdir, hcopy
from utils.torch_io import save as hdfs_torch_save
logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self,
                 serialization_dir: str = ".output") -> None:
        self._serialization_dir = serialization_dir
        if not hexists(self._serialization_dir):
            hmkdir(self._serialization_dir)

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        step: int = -1) -> None:
        """
        Save ckpt to local or HDFS
        """
        if step > 0:
            model_path = os.path.join(
                self._serialization_dir, "model_state_step_{}.th".format(step))
            hdfs_torch_save(model_state, model_path)

        else:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch))

            training_path = os.path.join(self._serialization_dir,
                                         "training_state_latest.th")
            hdfs_torch_save(model_state, model_path)
            hdfs_torch_save({**training_states, "epoch": epoch}, training_path)
