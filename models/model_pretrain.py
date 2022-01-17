import torch
from models import XVLMBase


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True, config_text=None)

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None, ret_bbox_loss=False):

        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        else:
            image_embeds, image_atts = self.get_vision_embeds(image)

        text_embeds = self.get_text_embeds(text_ids, text_atts)

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat)
        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        if ret_bbox_loss:
            output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)

            return loss_itc, loss_itm, loss_mlm, loss_bbox, loss_giou

        return loss_itc, loss_itm, loss_mlm
