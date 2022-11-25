# X-VLM: learning multi-grained vision language alignments


**[Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts](https://arxiv.org/abs/2111.08276). Yan Zeng, Xinsong Zhang, Hang Li. arXiv 2021.**

- Nov 2022: Release [X<sup>2</sup>-VLM](https://arxiv.org/abs/2211.12402): All-In-One for Vision Language Tasks; All-In-One == Image + Video + Transfer to Other Languages / Domains
- May 2022: The paper has been accepted by ICML 2022 
- Jan 2022: Release official PyTorch implementation and X-VLM checkpoints
- Nov 2021: Release preprint in [arXiv](https://arxiv.org/abs/2111.08276)


X-VLM (216M parameters: swin-base + 6L text + 6L cross):
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/cross-modal-retrieval-on-coco-2014)](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/cross-modal-retrieval-on-flickr30k)](https://paperswithcode.com/sota/cross-modal-retrieval-on-flickr30k?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-grounding-on-refcoco-val)](https://paperswithcode.com/sota/visual-grounding-on-refcoco-val?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-grounding-on-refcoco-testa)](https://paperswithcode.com/sota/visual-grounding-on-refcoco-testa?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-grounding-on-refcoco-test-b)](https://paperswithcode.com/sota/visual-grounding-on-refcoco-test-b?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-reasoning-on-nlvr2-dev)](https://paperswithcode.com/sota/visual-reasoning-on-nlvr2-dev?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-reasoning-on-nlvr2-test)](https://paperswithcode.com/sota/visual-reasoning-on-nlvr2-test?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/visual-question-answering-on-vqa-v2-test-dev)](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2-test-dev?p=multi-grained-vision-language-pre-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-grained-vision-language-pre-training/image-captioning-on-coco-captions)](https://paperswithcode.com/sota/image-captioning-on-coco-captions?p=multi-grained-vision-language-pre-training)

<div align="center">
  <img width="70%" src="x-vlm-results.png">
</div>


## Hiring
We are looking for interns / FTEs at ByteDance AI-LAB (in Beijing / Shanghai)! If you are interested in working with us on vision language models, please send your resume to 
zhangxinsong.0320@bytedance.com.


## Features
- Support several backbones 
    - vision encoder: deit / clip-vit / swin-transformer 
    - text encoder: bert / roberta
- Support apex O1 / O2 for pre-training
- Read from and write to HDFS
- Distributed training across nodes for both pre-training and fine-tuning

Please read the code for more details. 


## Requirements
- Install python3 environment
```angular2html
pip3 install -r requirements.txt
```
- Download raw images from corresponding websites
- Download the json files we provided, which contains image read paths and captions and/or bbox annotations
- If running pre-training scripts: 
  - install Apex
  - download pre-trained models for parameter initialization 
    - image encoder: [clip-vit-base](https://huggingface.co/openai/clip-vit-base-patch16/tree/main) / [swin-transformer-base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)
    - text encoder: [bert-base](https://huggingface.co/bert-base-uncased/tree/main)
- Organize these files like this (% is for pre-training only):  
```angular2html
X-VLM/
    data/
        finetune/
            refcoco+/*.json
            *.json
        
        %pretrain_4m/*.json
        %swin_base_patch4_window7_224_22k.pth
        %bert-base-uncased/
            config.json
            pytorch_model.bin
            tokenizer_config.json
            tokenizer.json
            vocab.txt

    images/
        coco/
            train2014/*.jpg
            val2014/*.jpg
            test2015/*.jpg
        
        visualgenome/
            image/*.jpg
        
        nlvr2/
            images/
                train/0-99/*.png
            dev/*.png
            test1/*.png
        
        %sbu/*.jpg
        %cc-3m/*.jpg
```


## Pretrain
```angular2html
python3 run.py --task "pretrain_4m_base" --dist "1" --output_dir "output/pretrain_4m_base"
```
For distributed training across nodes, see run.py for more details. To make a fair comparison of some recent works, we pre-trained X-VLM (4M/16M) for 200K steps. 

#### Data

<div align="center">
  <img width="50%" src="x-vlm-pretrain-data-1.png">
</div>

<div align="center">
  <img width="50%" src="x-vlm-pretrain-data-2.png">
</div>

***
🌟UPDATE: our multi-lingual multi-modal project [Cross-View Language Modeling](https://github.com/zengyan-97/CCLM) released the text of [COCO+VG+SBU+CC3M](https://github.com/zengyan-97/CCLM) and  [Object And Region Annotations](https://github.com/zengyan-97/CCLM) in six languages. You can use english text for X-VLM pre-training. 
***

All datasets we utilized are publicly available. We cannot re-distribute the data. So, please prepare the pre-training data by yourself. Here, we provide some data examples. Read the code dataset/pretrain_dataset.py/ImageTextJsonDataset & RegionTextJsonDataset for details.  
```
# image-captions pairs, providing 'binary' or 'image_rpath' 
{'caption': 'dog on bike in harajuku', 
 'binary': binary_encoding_of_the_image, 
 'image_rpath': local_rpath_of_the_image
}


# object/region annotations, providing 'binary' or 'image_rpath' 
{'elems': [{'caption': 'lady sitting at table that has pizza on it',  # str or list of str  
            'bb': [155, 0, 205, 131]   # (x, y, w, h)
            }, 
           {'caption': 'window',  
            'attributes': 'closed',  # str or list of str 
            'bb': [20, 130, 335, 185]
            },
          ]
 'caption': if_exist,  # str or list of str 
 'binary': binary_encoding_of_the_image, 
 'image_rpath': local_rpath_of_the_image
}
```



#### Checkpoints
[X-VLM (4M, 200K steps)](https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing)  
[X-VLM (16M, 200K steps)](https://drive.google.com/file/d/1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0/view?usp=sharing)


## Finetune
Datasets for finetuning and checkpoints of X-VLM (4M/16M) can be downloaded in following links. 

#### Data 
[download json files](https://drive.google.com/file/d/1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4/view?usp=sharing) 


#### Checkpoints and Logs (16M)
[retrieval-mscoco](https://drive.google.com/drive/folders/1VotCNmdevvtMuJmdxPfg3MOZXJRnV96D?usp=sharing)  
[retrieval-flickr](https://drive.google.com/drive/folders/1lsuBVP7MEqGqWkqRxaxb8N8TbSKqQ1Yz?usp=sharing)  
[vqa](https://drive.google.com/drive/folders/1tRKlCVMvkRquad7kMp4JVEbaKG-Ho8To?usp=sharing)  
[nlvr2](https://drive.google.com/drive/folders/19Vz9h0oDRcbinUIcbfh-dsNwtlzrkQiP?usp=sharing)  
[refcoco](https://drive.google.com/drive/folders/1ySQTjpTm5CeHp50YYFObUjT7DTHLN7DZ?usp=sharing)  
[refcoco-weak](https://drive.google.com/drive/folders/1wvpsA-VONdDUwQdQITG-V7Kc2CkIkbfM?usp=sharing)  
[captioning-coco](https://drive.google.com/drive/folders/15Ymsay477QKo3PWOt9cwjWpiII5RQaH8?usp=sharing)  

#### Checkpoints and Logs (4M)
[4m-all-ft-ckpts.tar](https://drive.google.com/file/d/1laNJHBnVGF7onbEYh1vO-b2P5TxdqH-k/view?usp=sharing)



#### Examples
```angular2html
# train
python3 run.py --task "vqa" --dist "1" --output_dir "output/vqa" --checkpoint "4m_base_model_state_step_199999.th"

# train: if using >2 nodes for fine-tuning, specify --output_hdfs to save some tmp results; it is only required by vqa & refcoco 
python3 run.py --task "vqa" --dist "all" --output_dir "output/vqa" --output_hdfs "hdfs://xxx/vqa_tmp" --checkpoint "4m_base_model_state_step_199999.th"  

# evaluate
python3 run.py --task "vqa" --dist "1" --evaluate --output_dir "output/vqa_eval" --checkpoint "4m_base_finetune/vqa/model_state_epoch_9.th"
```
Specify "--task" to finetune on **image-text retrieval, nlvr2, visual grounding, or image captioning**. See run.py for details.


#### More Examples of Captioning:
```angular2html
# adapt cross-modal encoder + MLM head -> lm decoder; subsequent fine-tuning is included   
python3 run.py --task "coco_capt_domain" --dist "1" --output_dir "output/coco_capt_domain" --checkpoint "4m_base_model_state_step_199999.th"

# fine-tune only; evaluate is included 
python3 run.py --task "coco_captioning" --dist "1" --output_dir "output/coco_captioning" --checkpoint "4m_base_finetune/coco_caption/lm_domain_pretrain.th"
# evaluate only
python3 run.py --task "coco_captioning" --dist "1" --output_dir "output/coco_captioning" --evaluate --checkpoint "4m_base_finetune/coco_caption/coco_capt_ft_epoch_4.th"

# further CIDEr optimization; evaluate is included 
python3 run.py --task "coco_captioning_scst" --dist "1" --output_dir "output/coco_captioning_scst" --checkpoint "4m_base_finetune/coco_caption/coco_capt_ft_epoch_4.th"
# evaluate only
python3 run.py --task "coco_captioning" --dist "1" --output_dir "output/coco_captioning_scst" --evaluate --checkpoint "4m_base_finetune/coco_caption/coco_capt_cider_step_41000.th"
```

To make a fair comparison, we follow the previous works for fine-tuning. So, some scripts are based on ALBEF, OSCAR, and BLIP. We thank the authors for opening source their code.


#### Evaluation on VLUE
[VLUE](https://github.com/MichaelZhouwang/VLUE) is a new OOD benchmark to evaluate vision-language models, which has been accepted by ICML2022.   
```angular2html
python3 run.py --task "eval_vlue_itr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "itr_coco/checkpoint_9.pth"

python3 run.py --task "eval_vlue_vqa" --dist "1" --evaluate  --output_dir "output/" --checkpoint "vqa/model_state_epoch_9.th"

python3 run.py --task "eval_vlue_nlvr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "nlvr/nlvr_ft/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco_bbox/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco_weakly" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco/checkpoint_best.pth"
```

## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{xvlm,
  title={Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts},
  author={Zeng, Yan and Zhang, Xinsong and Li, Hang},
  journal={arXiv preprint arXiv:2111.08276},
  year={2021}
}
```


### Contact
For issues using this code, please submit a GitHub issue.