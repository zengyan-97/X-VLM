import os
import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from dataset.utils import pre_caption


class coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt=''):
        self.annotation = []
        for f in ann_rpath:
            self.annotation += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class coco_karpathy_train_scst(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt=''):
        self.annotation = []
        self.image_captions_map = {}

        for f in ann_rpath:
            for ann in json.load(open(f, 'r')):
                self.annotation.append(ann)

                if ann['image'] in self.image_captions_map.keys():
                    self.image_captions_map[ann['image']].append(ann['caption'])
                else:
                    self.image_captions_map[ann['image']] = [ann['caption']]

        counter = Counter()
        for _, v in self.image_captions_map.items():
            counter[len(v)] += 1
        print("### image_captions_map, ", counter, flush=True)

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # w/o prompt
        captions_gt = [pre_caption(c, self.max_words) for c in self.image_captions_map[ann['image']]]

        return image, random.sample(captions_gt, 5)

    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        image_list, captions_gt_list = batch

        images = torch.stack(image_list)

        return images, captions_gt_list


class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_rpath, split):
        self.annotation = json.load(open(ann_rpath, 'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)
