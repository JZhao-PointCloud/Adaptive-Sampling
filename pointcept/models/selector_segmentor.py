import torch
import torch.nn as nn
from pointcept.models.builder import MODELS, build_model
from .selectors import build_selector


@MODELS.register_module("SelectorSegmentor-v1m1")
class SelectorSegmentor(nn.Module):
    """Fixed sampled indices followed by an independently trained segmentor."""

    def __init__(self, selector, segmentor, ignore_index=-1, eval_map_to_full=True):
        super().__init__()
        self.selector = build_selector(selector)
        self.segmentor = build_model(segmentor)
        self.ignore_index = ignore_index
        self.criteria = None
        self.eval_map_to_full = eval_map_to_full

    def _compute_seg_loss(self, seg_logits, segment):
        if seg_logits.numel() == 0:
            return seg_logits.sum() * 0.0
        valid_mask = segment != self.ignore_index
        if not valid_mask.any():
            return seg_logits.sum() * 0.0
        if self.criteria is not None:
            return self.criteria(seg_logits, segment)
        if hasattr(self.segmentor, "criteria"):
            return self.segmentor.criteria(seg_logits, segment)
        raise ValueError("No segmentation criteria available for SelectorSegmentor")

    def _lift_logits_to_full(
        self, seg_logits, sampled_input, full_input, k=1, idw=False
    ):
        import pointops

        k = max(1, min(int(k), int(seg_logits.shape[0])))
        reverse, distance = pointops.knn_query(
            k,
            sampled_input["coord"].float(),
            sampled_input["offset"].int(),
            full_input["coord"].float(),
            full_input["offset"].int(),
        )
        reverse = reverse.reshape(-1, k).long()
        if k == 1 or not idw:
            return seg_logits[reverse[:, 0]]
        distance = distance.reshape(-1, k).to(seg_logits.dtype)
        inv_distance = 1.0 / distance.clamp_min(1e-08)
        exact_mask = distance <= 1e-08
        if exact_mask.any():
            exact_weight = exact_mask.to(seg_logits.dtype)
            has_exact = exact_mask.any(dim=1, keepdim=True)
            weight = torch.where(has_exact, exact_weight, inv_distance)
        else:
            weight = inv_distance
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-06)
        gathered = seg_logits[reverse]
        return (gathered * weight.unsqueeze(-1)).sum(dim=1)

    def forward(self, input_dict, return_point=False):
        selection = self.selector(input_dict)
        sampled = selection.sampled_input
        output = self.segmentor(
            sampled, return_point=return_point, return_logits=not self.training
        )
        result = dict(selection.aux_metrics)
        result.update(output)
        if self.training:
            return result
        sparse = output["seg_logits"]
        result.update(
            eval_sparse_logits=sparse,
            eval_sparse_segment=sampled.get("segment"),
            eval_sparse_coord=sampled["coord"],
            eval_sparse_offset=sampled["offset"],
        )
        if self.eval_map_to_full:
            result["seg_logits"] = self._lift_logits_to_full(
                sparse, sampled, input_dict
            )
            if "segment" in input_dict:
                result["loss"] = self._compute_seg_loss(
                    result["seg_logits"], input_dict["segment"]
                )
        return result
