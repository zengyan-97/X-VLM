# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import os
import sys
import time
import random
import argparse

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

############ Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 8  # e.g. 8 gpus

MASTER_ADDR = 'SET_IT'
MASTER_PORT = 12345
NODE_RANK = 0  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_nnodes(args):  # when using only part of nodes
    if args.dist == 'all':
        return NNODES
    else:
        return 1


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:}".format(
            NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes=1 ".format(NPROC_PER_NODE)

    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_pretrain(args):
    print("### Start pre-training", flush=True)
    dist_launch = get_dist_launch(args)
    os.system(f"{dist_launch} --use_env Pretrain.py --config {args.config} --output_dir {args.output_dir}")


def run_pretrain_nlvr(args):
    print("### Start nlvr domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        args.checkpoint = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = 'configs/NLVR_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env NLVR_pretrain.py --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        args.checkpoint = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    # run fine-tune
    if len(args.output_dir): args.output_dir += '_nlvr2'
    run_nlvr2(load_nlvr_pretrain=True)


def run_pretrain_refcoco_bbox():
    print("### Start refcoco bbox domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        args.checkpoint = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = './configs/Grounding_bbox_pretrain_O1.yaml'

        os.system(f"{dist_launch} "
                  f"--use_env Grounding_bbox_pretrain.py --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        args.checkpoint = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    # run fine-tune
    if len(args.output_dir): args.output_dir += '_refcoco'
    run_refcoco(use_bbox=True, load_bbox_pretrain=True)


def run_nlvr2(load_nlvr_pretrain=False):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/nlvr2")

    print("### Training NLVR2", flush=True)
    os.system(f"{dist_launch} "
              "--use_env NLVR.py --config configs/NLVR.yaml "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--load_nlvr_pretrain' if load_nlvr_pretrain else ''} "
              f"{'--evaluate' if args.evaluate else ''}")


def run_itr_flickr():
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/flickr30k-images")

    print("### Training Retrieval Flickr", flush=True)
    os.system(f"{dist_launch} "
              "--use_env Retrieval.py --config configs/Retrieval_flickr.yaml "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_itr_coco():
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training Retrieval COCO", flush=True)
    os.system(f"{dist_launch} "
              "--use_env Retrieval.py --config configs/Retrieval_coco.yaml "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_vqa(args):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco") and os.path.exists("images/visualgenome")

    print("### Training VQA", flush=True)
    if not os.path.exists(args.config): args.config = './configs/VQA.yaml'

    os.system(f"{dist_launch} "
              f"--use_env VQA.py --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run_refcoco(use_bbox=False, block_num=-1, load_bbox_pretrain=False, epochs=-1):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    if use_bbox:
        print("### Training RefCOCO with bbox", flush=True)

        os.system(f"{dist_launch} "
                  "--use_env Grounding_bbox.py --config configs/Grounding_bbox.yaml "
                  f"--output_dir {args.output_dir} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
                  f"--bs {args.bs} {'--load_bbox_pretrain' if load_bbox_pretrain else ''} --checkpoint {args.checkpoint} "
                  f"{'--evaluate' if args.evaluate else ''}")

    else:
        print("### Training RefCOCO", flush=True)
        os.system(f"{dist_launch} "
                  f"--use_env Grounding.py --config ./configs/Grounding.yaml "
                  f"--output_dir {args.output_dir} --bs {args.bs} {f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} "
                  f"--gradcam_mode itm --block_num {block_num} --epochs {epochs} --checkpoint {args.checkpoint} "
                  f"{'--evaluate' if args.evaluate else ''}")


def run_pretrain_captioning(args):
    print("### Start captioning domain pre-training", flush=True)

    dist_launch = get_dist_launch(args)

    if len(args.load_ckpt_from):
        print(f"### Loading domain pre-trained results from: {args.load_ckpt_from}")
        domain_ckpt = get_from_hdfs(args.load_ckpt_from)

    else:  # domain pre-train
        if not os.path.exists(args.config): args.config = f'configs/{args.model}/Captioning_pretrain_O1.yaml'

        os.system(f"{dist_launch} --use_env Captioning_pretrain.py --seed {args.seed} --config {args.config} "
                  f"--output_dir {args.output_dir} --checkpoint {args.checkpoint}")

        domain_ckpt = get_from_hdfs(f"{args.output_dir}/model_state_epoch_latest.th")

    return domain_ckpt


def run_coco_captioning(args, load_capt_pretrain=False, scst=False):
    dist_launch = get_dist_launch(args)

    assert os.path.exists("images/coco")

    print("### Training COCO Captioning", flush=True)

    if not os.path.exists(args.config):
        args.config = f'./configs/{args.model}/Captioning.yaml'

    if scst:
        load_capt_pretrain = True  # same way to load ckpt;

    os.system(f"{dist_launch} "
              f"--use_env {'Captioning_scst.py' if scst else 'Captioning.py'} --config {args.config} "
              f"{f'--output_hdfs {args.output_hdfs}' if len(args.output_hdfs) else ''} --output_dir {args.output_dir} "
              f"--bs {args.bs} --seed {args.seed} --checkpoint {args.checkpoint} "
              f"{'--scst' if scst else ''}  {'--load_capt_pretrain' if load_capt_pretrain else ''} {'--evaluate' if args.evaluate else ''}")


def run(args):
    if args.task not in ['pretrain_4m_base']:
        assert hexists(args.checkpoint) or hexists(args.load_ckpt_from)

    if args.task == 'pretrain_4m_base':
        args.config = 'configs/Pretrain_XVLM_base_4m.yaml'
        run_pretrain(args)

    elif args.task == 'itr_2_tasks':
        run_itr_coco()
        run_itr_flickr()

    elif args.task == 'itr_coco':
        run_itr_coco()

    elif args.task == 'itr_flickr':
        run_itr_flickr()

    elif args.task == 'vqa':
        run_vqa(args)

    elif args.task == 'vqa_480':
        # if use 480x480 (the accuracy will increase 0.5~1%):
        args.config = "configs/VQA_480.yaml"
        run_vqa(args)

    elif args.task == 'nlvr':
        run_pretrain_nlvr(args)
        # run_nlvr2()  # w/o domain pretrain

    elif args.task == 'refcoco_weakly':
        run_refcoco(block_num=9)  # 9 for X-VLM base

    elif args.task == 'refcoco_block_num_search':  # for refcoco_weakly
        # block_num: use which layer of the cross-modal encoder for calculation
        # it is a critical hyper-param for refcoco without bbox annotations
        for num in [8, 9, 10, 7]:
            print(f"### block_num {num}")
            run_refcoco(block_num=num, epochs=1)

    elif args.task == 'refcoco_bbox':
        run_pretrain_refcoco_bbox()
        # run_refcoco(use_bbox=True)  # w/o domain pretrain

    elif args.task.startswith('coco_capt_domain'):
        if args.task == 'coco_capt_domain_14m':
            args.config = f'configs/{args.model}/Captioning_pretrain_O1_14m.yaml'

        domain_ckpt = run_pretrain_captioning(args)

        # run fine-tune, reset args
        args.checkpoint = domain_ckpt
        if hexists(args.output_dir): args.output_dir = os.path.join(args.output_dir, 'coco_capt_ft')
        args.config = f'./configs/{args.model}/Captioning.yaml'
        run_coco_captioning(args, load_capt_pretrain=True)

    elif args.task == 'coco_captioning':
        run_coco_captioning(args, load_capt_pretrain=True)

    elif args.task == 'coco_captioning_scst':  # load checkpoint of 'coco_captioning' results
        args.config = f'./configs/{args.model}/Captioning_scst.yaml'
        run_coco_captioning(args, scst=True)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")

    parser.add_argument('--config', default='', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")

    parser.add_argument('--checkpoint', default='', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default='', type=str, help="load domain pre-trained params")

    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, required=True, help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--output_hdfs', type=str, default='', help="HDFS path required by VQA and Refcoco, "
                                                                    "to collect eval results among nodes")

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")

    args = parser.parse_args()

    if MASTER_ADDR == 'SET_IT':
        print("### warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    if '/SET/PATH/TO/hadoop/bin/hdfs' in HADOOP_BIN:
        print("### warning: you have not set the path to hadoop_bin (ignore this if you don't use HDFS)")

    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)

    if len(args.output_hdfs):
        assert hexists(os.path.dirname(args.output_hdfs))

    if len(args.config):
        assert hexists(args.config)

        if args.config.startswith('hdfs://'):
            args.config = get_from_hdfs(args.config)

    if args.checkpoint.startswith('hdfs://'):
        args.checkpoint = get_from_hdfs(args.checkpoint)

    run(args)

