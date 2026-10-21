import torch
import torch.nn as nn
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


def _normalize_input_grid_size(input_dict):
    if "grid_size" not in input_dict:
        return input_dict
    grid_size = input_dict["grid_size"]
    if torch.is_tensor(grid_size) and grid_size.numel() > 1:
        input_dict = dict(input_dict)
        input_dict["grid_size"] = grid_size.reshape(-1)[0]
    return input_dict


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        freeze_seg_head=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.freeze_seg_head = freeze_seg_head
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.freeze_seg_head:
            for p in self.seg_head.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False, return_logits=False):
        input_dict = _normalize_input_grid_size(input_dict)
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            if return_logits:
                return_dict["seg_logits"] = seg_logits
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
