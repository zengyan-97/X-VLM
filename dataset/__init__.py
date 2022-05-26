import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.re_dataset import re_train_dataset, re_eval_dataset
from dataset.pretrain_dataset import ImageTextJsonDataset, RegionTextJsonDataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset, grounding_dataset_bbox
from dataset.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_train_scst, coco_karpathy_caption_eval


from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, evaluate=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform)

        region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                transform=pretrain_transform, box_transform=box_transform)

        return general_dataset, region_dataset

    elif dataset == 're':
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, None, test_dataset

        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'vqa':
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'],
                                       text_encoder=config['text_encoder'], use_roberta=config['use_roberta'])
        if evaluate:
            return None, vqa_test_dataset

        train_dataset = vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'], config['vg_root'],
                                    split='train', text_encoder=config['text_encoder'], use_roberta=config['use_roberta'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'nlvr_pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform)

        return general_dataset

    elif dataset == 'nlvr':
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, None, test_dataset

        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'grounding':
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        return train_dataset, test_dataset

    elif dataset == 'grounding_bbox_pretrain':
        region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                transform=pretrain_transform, box_transform=box_transform)

        return region_dataset

    elif dataset == 'grounding_bbox':
        test_dataset = grounding_dataset_bbox(config['test_file'], test_transform, config['image_root'], mode='test', config=config)
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset_bbox(config['train_file'], train_transform, config['image_root'], mode='train', config=config)
        return train_dataset, test_dataset

    elif dataset == 'captioning_pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform, add_eos=True)
        return general_dataset

    elif dataset == 'caption_coco':
        train_dataset = coco_karpathy_train(train_transform, config['image_root'], config['train_file'], prompt=config['prompt'], max_words=config['max_tokens'])
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    elif dataset == 'caption_coco_scst':
        train_dataset = coco_karpathy_train_scst(train_transform, config['image_root'], config['train_file'],
                                            prompt=config['prompt'], max_words=config['max_tokens'])
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
