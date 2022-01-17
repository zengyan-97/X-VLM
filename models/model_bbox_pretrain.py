import torch
from models import XVLMBase, load_pretrained


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=True)
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=False, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx_to_group_img, target_bbox, is_image=None):
        image_embeds_fullatts, _ = self.get_vision_embeds(image, idx_to_group_img=idx_to_group_img)
        text_embeds = self.get_text_embeds(text_ids, text_atts)

        output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
        loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

        return loss_bbox, loss_giou