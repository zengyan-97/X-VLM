# Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts (https://arxiv.org/abs/2111.08276)
# Github: https://github.com/zengyan-97/X-VLM
# Copyright (c) 2022, ByteDance Inc.
# All rights reserved.

import json
import copy
import math
import random
import sys
import re
import io
import traceback
from base64 import b64decode

from random import randint, shuffle
from random import random as rand

import torch
from torchvision.transforms.functional import hflip, resize
from transformers import BertTokenizer, RobertaTokenizer

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
from dataset.dist_dataset import DistLineReadingDataset


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token
        print("mask_generator.cls_token, ", self.cls_token, flush=True)
        print("mask_generator.mask_token, ", self.mask_token, flush=True)

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


class ImageTextJsonDataset(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, add_eos=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        if 'images' in config.keys():
            self.image_key = config['images']['image_key']
            self.is_image_rpath = config['images']['is_image_rpath']
            self.caption_key = config['images']['caption_key']
            self.batch_size = config['images']['batch_size']
            self.tokenized = config['images']['tokenized']

        self.use_roberta = config['use_roberta']
        self.tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder']) if self.use_roberta else \
            BertTokenizer.from_pretrained(config['text_encoder'])

        self.add_eos = add_eos

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        print("dataset.cls_token, ", self.cls_token, flush=True)
        print("dataset.eos_token, ", self.eos_token, flush=True)
        print("dataset.pad_token_id, ", self.pad_token_id, flush=True)
        print("dataset.mask_token_id, ", self.mask_token_id, flush=True)

        self.mask_generator = TextMaskingGenerator(self.tokenizer, config['mask_prob'],
                                                   config['max_masks'], config['skipgram_prb'],
                                                   config['skipgram_size'], config['mask_whole_word'])

        self.PAD_mask = -100  # loss will ignore this
        self.max_words = config['max_words']
        self.max_tokens = config['max_tokens']
        self.max_masks = config['max_masks']

        self.transform = transform
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                caption = ann[self.caption_key]
                if isinstance(caption, list):
                    caption = random.choice(caption)

                if self.is_image_rpath:  # read path or base64 encoding
                    image = Image.open(ann[self.image_key]).convert('RGB')
                else:
                    # if reading from HDFS, use this:
                    image = Image.open(io.BytesIO(b64decode(ann[self.image_key]))).convert("RGB")

                image = self.transform(image)

                text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                yield image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()

    def preprocess(self, text):
        if self.tokenized:
            tokens = text.strip().split(' ')
        else:
            text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
            tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        if self.add_eos:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids

    def collate_fn(self, batch):
        batch_tensors = []
        for x in zip(*batch):
            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors


class RegionTextJsonDataset(ImageTextJsonDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, box_transform=None):
        super().__init__(config, data_path, rank=rank, world_size=world_size, shuffle=shuffle,
                         repeat=repeat, transform=transform)

        self.image_key = config['regions']['image_key']
        self.is_image_rpath = config['regions']['is_image_rpath']
        self.caption_key = config['regions']['caption_key']
        assert self.caption_key == 'caption', "please follow my data format"
        self.batch_size = config['regions']['batch_size']
        self.tokenized = config['regions']['tokenized']
        self.careful_hflip = config['regions']['careful_hflip'] if 'careful_hflip' in config['regions'] else False

        self.box_transform = box_transform
        self.max_regions = config['regions']['max_regions']
        self.min_perc_in_image = config['regions']['min_perc_in_image']

    def get_bbox(self, ann):
        x, y, w, h = ann['bb']
        return int(x), int(y), int(w), int(h)

    def left_or_right_in_caption(self, ann):
        def _in_it(elem):
            if isinstance(elem['caption'], list):
                for caption in elem['caption']:
                    if ('left' in caption) or ('right' in caption):
                        return True
            else:
                if ('left' in elem['caption']) or ('right' in elem['caption']):
                    return True

        if 'caption' in ann.keys():
            if _in_it(ann):
                return True

        for elem in ann['elems']:
            if _in_it(elem):
                return True

        return False

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                assert isinstance(ann, dict), "ann is not dict"

                try:
                    image = Image.open(ann[self.image_key]).convert('RGB') if self.is_image_rpath \
                        else Image.open(io.BytesIO(b64decode(ann[self.image_key]))).convert("RGB")
                except Warning:
                    raise ValueError("### Warning: RegionTextJsonDataset Image.open")

                W, H = image.size

                # random crop
                x, y, w, h = self.get_bbox(random.choice(ann['elems']))
                assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (h > 0), "elem invalid"

                x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
                x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H), H)
                w0, h0 = x1 - x0, y1 - y0
                assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (h0 > 0), "elem randomcrop, invalid"

                image = image.crop((x0, y0, x0 + w0, y0 + h0))
                W, H = image.size

                do_hflip = False
                if rand() < 0.5:
                    if self.careful_hflip and self.left_or_right_in_caption(ann):
                        pass
                    else:
                        image = hflip(image)
                        do_hflip = True

                image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
                image = self.box_transform(image)

                text_ids_list = []
                text_ids_masked_list = []
                text_atts_list = []
                masked_pos_list = []
                masked_ids_list = []
                image_atts_list = []

                target_bbox_list = []
                is_image_list = []

                max_elems = self.max_regions

                if 'caption' in ann.keys():
                    caption = random.choice(ann['caption']) if isinstance(ann['caption'], list) else ann['caption']
                    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

                    text_ids_list.append(text_ids)
                    text_atts_list.append(text_atts)
                    text_ids_masked_list.append(text_ids_masked)
                    masked_pos_list.append(masked_pos)
                    masked_ids_list.append(masked_ids)

                    image_atts_list.append([1] * (self.num_patch ** 2 + 1))
                    target_bbox_list.append(torch.tensor([0.5, 0.5, 1, 1], dtype=torch.float))
                    is_image_list.append(1)

                    max_elems -= 1

                elems = random.sample(ann['elems'], len(ann['elems']))

                for elem in elems:
                    if max_elems <= 0:
                        break

                    x, y, w, h = self.get_bbox(elem)

                    xx, yy = max(x0, x), max(y0, y)
                    xm, ym = min(x0 + w0, x + w), min(y0 + h0, y + h)
                    if (xm > xx) and (ym > yy):
                        if (xm - xx) * (ym - yy) / (w * h) > self.min_perc_in_image:
                            x, y, w, h = xx, yy, xm - xx, ym - yy  # part inside the cropped image

                            # axis transform: after crop
                            x = x - x0
                            y = y - y0

                            if do_hflip:  # flipped applied
                                x = (W - x) - w  # W is w0

                            # resize applied
                            x = self.image_res / W * x
                            w = self.image_res / W * w
                            y = self.image_res / H * y
                            h = self.image_res / H * h

                            caption = random.choice(elem['caption']) if isinstance(elem['caption'], list) else elem['caption']

                            if 'attributes' in elem.keys():
                                elem_attr = random.choice(elem['attributes']) if isinstance(elem['attributes'], list) else elem['attributes']
                                caption = elem_attr + ' ' + caption

                            text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)
                            image_atts = self.get_image_attns(x, y, w, h)

                            text_ids_list.append(text_ids)
                            text_atts_list.append(text_atts)
                            text_ids_masked_list.append(text_ids_masked)
                            masked_pos_list.append(masked_pos)
                            masked_ids_list.append(masked_ids)
                            image_atts_list.append(image_atts)

                            center_x = x + 1 / 2 * w
                            center_y = y + 1 / 2 * h

                            target_bbox_list.append(torch.tensor([center_x / self.image_res, center_y / self.image_res,
                                                                  w / self.image_res, h / self.image_res],
                                                                 dtype=torch.float))

                            is_image_list.append(0)

                            max_elems -= 1

                image_list = [image] if len(text_ids_list) else []

                yield image_list, text_ids_list, text_atts_list, text_ids_masked_list, masked_pos_list, \
                      masked_ids_list, image_atts_list, target_bbox_list, is_image_list

            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-' * 20)
                sys.stdout.flush()

    def get_image_attns(self, x, y, w, h):
        x_min = min(math.floor(x / self.patch_size), self.num_patch - 1)
        x_max = max(x_min+1, min(math.ceil((x+w) / self.patch_size), self.num_patch))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch - 1)
        y_max = max(y_min+1, min(math.ceil((y+h) / self.patch_size), self.num_patch))  # exclude

        image_atts = [0] * (1 + self.num_patch ** 2)
        image_atts[0] = 1  # always include [CLS]
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch * i + j + 1
                assert (index > 0) and (index <= self.num_patch ** 2), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts

    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        images, batch = batch[0], batch[1:]

        idx_to_group_img = []
        img_idx = -1
        for sample in batch[0]:
            n_elems = len(sample)
            if n_elems > 0:
                img_idx += 1
                idx_to_group_img.extend([img_idx] * n_elems)  # flatten

        batch_size = self.batch_size
        n_elems = len(idx_to_group_img)
        to_keep = list(range(n_elems))
        if n_elems >= batch_size:
            to_keep = random.sample(to_keep, batch_size)
        else:
            # fixed batch_size is required. otherwise, the process will be blocked. so, i do pad here.
            # but pad causes wrong calculation for contrastive learning.
            # Set appropriate batch_size, max_images, and max_regions to avoid frequent padding.
            try:
                to_pad = random.sample(to_keep, batch_size - n_elems)
                to_keep += to_pad
                print("### warning: pad region_batch by sampling, ", len(to_pad), flush=True)

            except ValueError:
                print("### warning: pad region_batch by expanding, ", batch_size-len(to_keep), flush=True)
                to_keep = (to_keep * math.ceil(batch_size/len(to_keep)))[:batch_size]

        images = torch.stack(sum(images, []))  # flatten
        idx_to_group_img = torch.tensor([idx_to_group_img[index] for index in to_keep], dtype=torch.long)

        batch_tensors = [images, idx_to_group_img]
        for x in [sum(x, []) for x in batch]:

            x = [x[index] for index in to_keep]

            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors
