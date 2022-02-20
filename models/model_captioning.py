import copy
import torch
import torch.nn.functional as F

from transformers import BertTokenizer

from models.xbert import BertLMHeadModel
from models.xroberta import RobertaForCausalLM

from models import XVLMBase, load_pretrained


class XVLM(XVLMBase):  # for domain pretrain
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

        assert config['text_encoder'] == 'data/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained('data/bert-base-uncased')
        self.tokenizer.add_special_tokens({'bos_token': self.tokenizer.cls_token, 'eos_token': self.tokenizer.sep_token})

        self.prompt = config['prompt']
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        self.max_tokens = config['max_tokens']

        config_enc = self.text_encoder.config

        self.text_encoder = None
        self.text_decoder = BertLMHeadModel(config=config_enc, label_smoothing=config['label_smoothing'])

    def load_pretrained(self, ckpt_rpath, config, load_capt_pretrain=False, is_eval=False):
        if is_eval:
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)

        else:
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)
            print("### Loading pretrained text encoder", flush=True)
            print("load_capt_pretrain, ", load_capt_pretrain)
            if not load_capt_pretrain:
                print("### Loading pretrained text encoder", flush=True)
                for key in list(state_dict.keys()):
                    assert isinstance(key, str)
                    if key.startswith('text_encoder.'):
                        decoder_key = key.replace('text_encoder.', 'text_decoder.')
                        state_dict[decoder_key] = state_dict[key]
                        del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, caption):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.max_tokens, return_tensors="pt").to(
            image.device)

        # text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        loss = self.text_decoder(text.input_ids,
                                 attention_mask=text.attention_mask,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 labels=decoder_targets,
                                 return_dict=True,
                                 ).loss

        return loss

    def generate(self, image, sample=False, num_beams=1, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0, num_return_sequences=1, greedy=False):

        prompt = [self.prompt] * image.size(0)

        image_embeds = self.vision_encoder(image)

        if num_beams > 1:
            assert (sample is False) and (num_return_sequences == 1)
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        if num_return_sequences > 1:
            assert (sample is True) and (num_beams == 1)
            image_embeds = image_embeds.repeat_interleave(num_return_sequences, dim=0)
            prompt = [self.prompt] * image_embeds.size(0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        def _get_captions(caption_ids):
            captions = []
            for output in caption_ids:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption[len(self.prompt):])
            return captions

        if greedy:
            # greedy generation from OSCAR
            assert (num_beams == 1) and (num_return_sequences == 1)
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=False, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=image_embeds.size(0), **model_kwargs)

            return _get_captions(outputs)

        elif sample:
            # sampling from OSCAR
            outputs, logprobs = self.text_decoder._generate_no_beam_search(input_ids=input_ids, cur_len=input_ids.shape[1], max_length=max_length,
                                          do_sample=True, temperature=1,
                                          top_k=0, top_p=1, repetition_penalty=repetition_penalty,
                                          pad_token_id=self.tokenizer.pad_token_id, eos_token_ids=[self.tokenizer.sep_token_id],
                                          batch_size=image_embeds.size(0), **model_kwargs)

            # outputs: (bs x num_return_sequences, max_length)
            # logprobs: (bs x num_return_sequences,)

            return _get_captions(outputs), logprobs

        else:
            # beam search from huggingface
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

            return _get_captions(outputs)
