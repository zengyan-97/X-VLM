import json
import os
import math
import random
from random import random as rand

import torch
from torch.utils.data import Dataset

from torchvision.transforms.functional import hflip, resize

from PIL import Image
from dataset.utils import pre_caption
from refTools.refer_python3 import REFER


class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train'):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode

        if self.mode == 'train':
            self.img_ids = {}
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1            
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['text'], self.max_words)

        if self.mode == 'train':
            img_id = ann['image'].split('/')[-1]

            return image, caption, self.img_ids[img_id]
        else:
            return image, caption, ann['ref_id']


class grounding_dataset_bbox(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train', config=None):
        self.refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
        self.image_res = config['image_res']
        self.careful_hflip = config['careful_hflip']

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode

        if self.mode == 'train':
            self.img_ids = {}
            n = 0
            for ann in self.ann:
                img_id = ann['image'].split('/')[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

    def __len__(self):
        return len(self.ann)

    def left_or_right_in_caption(self, caption):
        if ('left' in caption) or ('right' in caption):
            return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = pre_caption(ann['text'], self.max_words)

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        W, H = image.size

        if self.mode == 'train':
            # random crop
            x, y, w, h = self.refer.refToAnn[ann['ref_id']]['bbox']
            assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (
                    h > 0), "elem invalid"

            x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
            x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H),
                                                                                 H)  # fix bug: max -> min
            w0, h0 = x1 - x0, y1 - y0
            assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (
                    h0 > 0), "elem randomcrop, invalid"
            image = image.crop((x0, y0, x0 + w0, y0 + h0))

            W, H = image.size

            do_hflip = False
            if rand() < 0.5:
                if self.careful_hflip and self.left_or_right_in_caption(caption):
                    pass
                else:
                    image = hflip(image)
                    do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
            image = self.transform(image)

            # axis transform: for crop
            x = x - x0
            y = y - y0

            if do_hflip:  # flipped applied
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            target_bbox = torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                        w / self.image_res, h / self.image_res], dtype=torch.float)

            return image, caption, target_bbox

        else:
            image = self.transform(image)  # test_transform
            return image, caption, ann['ref_id']