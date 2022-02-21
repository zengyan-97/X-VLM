# X-VLM: learning multi-grained vision language alignments

**[Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts](https://arxiv.org/abs/2111.08276). Yan Zeng, Xinsong Zhang, Hang Li. arXiv 2021.**

- Feb 2022: X-VLM supports image captioning  
- Jan 2022: release official PyTorch implementation and X-VLM checkpoints
- Nov 2021: release preprint in [arXiv](https://arxiv.org/abs/2111.08276)
 
  <img src="x-vlm-results.png" width="600">


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
For distributed training across nodes, see run.py for more details.


#### Data
Please prepare your own datasets. Read the code dataset/pretrain_dataset.py to see what format is needed. 

#### Checkpoints
[X-VLM (4M)](https://drive.google.com/file/d/1B3gzyzuDN1DU0lvt2kDz2nTTwSKWqzV5/view?usp=sharing)  
[X-VLM (16M)](https://drive.google.com/file/d/1VolF9P9cPSuD8CZMjwbKW20wUrAIaEFK/view?usp=sharing)


## Finetune
Datasets for finetuning and checkpoints of X-VLM (4M/16M) can be downloaded in following links. 

#### Data 
[download json files](https://drive.google.com/file/d/19SQGClFK9JnP6z4SH-EZ-xKsPQ3haPG5/view?usp=sharing) 


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

# if using >2 nodes for fine-tuning, specify --output_hdfs to save some tmp results.
python3 run.py --task "vqa" --dist "all" --output_dir "output/vqa" --output_hdfs "hdfs://xxx/vqa_tmp" --checkpoint "4m_base_model_state_step_199999.th"  

# evaluate
python3 run.py --task "vqa" --dist "1" --evaluate --output_dir "output/vqa_eval" --checkpoint "4m_base_finetune/vqa/model_state_epoch_9.th" 
```
Specify "--task" to finetune on **image-text retrieval, nlvr2, visual grounding, or image captioning**. See run.py for details.

Some fine-tuning scripts are based on ALBEF, OSCAR, and BLIP. We thank the authors for opening source their code.


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
For issues or help using this code, please submit a GitHub issue.