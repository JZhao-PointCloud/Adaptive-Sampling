import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcept.models.builder import MODELS, build_model


def marginal_probability_mse_loss(
    pred_prob,
    target_prob,
    *,
    valid_mask=None,
    sample_weight=None,
):
    pred_prob = pred_prob.float()
    target_prob = target_prob.to(device=pred_prob.device, dtype=torch.float32)
    target_prob = target_prob.clamp(0.0, 1.0)
    if valid_mask is None:
        valid_mask = torch.isfinite(target_prob)
    else:
        valid_mask = valid_mask.to(device=pred_prob.device, dtype=torch.bool)
        valid_mask = valid_mask & torch.isfinite(target_prob)

    if not valid_mask.any():
        return pred_prob.sum() * 0.0

    loss = F.mse_loss(pred_prob[valid_mask], target_prob[valid_mask], reduction="none")
    if sample_weight is None:
        return loss.mean()

    sample_weight = sample_weight.to(device=pred_prob.device, dtype=pred_prob.dtype)
    current_weight = sample_weight[valid_mask]
    return (loss * current_weight).sum() / current_weight.sum().clamp_min(1e-12)


class SamplerGenericBackbone(nn.Module):
    """Generic per-point sampler wrapper for non-PTV backbones.

    The backbone must return one feature vector per input point. This matches
    segmentation backbones such as SpUNet configured with ``num_classes`` as an
    intermediate feature width.
    """

    def __init__(
        self,
        backbone,
        out_channels,
        head_hidden_channels=64,
        head_num_layers=2,
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("sampler_backbone is required")
        if out_channels is None:
            raise ValueError(
                "sampler_backbone_out_channels is required for generic sampler"
            )
        from .learned import SamplerMLP

        self.backbone = build_model(dict(backbone))
        self.head = SamplerMLP(
            in_channels=int(out_channels),
            hidden_channels=int(head_hidden_channels),
            num_layers=int(head_num_layers),
        )

    @staticmethod
    def _prepare_input(input_dict):
        point_input = dict(input_dict)
        if "grid_size" in point_input and torch.is_tensor(point_input["grid_size"]):
            if point_input["grid_size"].numel() > 1:
                point_input["grid_size"] = point_input["grid_size"].reshape(-1)[0]
        return point_input

    def forward(self, input_dict, return_feat=False):
        point_or_feat = self.backbone(self._prepare_input(input_dict))
        feat = point_or_feat.feat if hasattr(point_or_feat, "feat") else point_or_feat
        score = self.head(feat)
        if return_feat:
            return score, feat
        return score


@MODELS.register_module("MarginalSamplerRegressor")
class MarginalSamplerRegressor(nn.Module):
    """Paper sampler: regress oracle keep probabilities with point-wise MSE."""

    def __init__(
        self,
        sampler_backbone,
        sampler_type="ptv3",
        sampler_backbone_out_channels=None,
        sampler_hidden_channels=96,
        sampler_num_layers=2,
        sampler_fusion_mode="concat",
        target_key="marginal_prob",
        ignore_index=-1,
        mask_ignore_index=True,
    ):
        super().__init__()
        self.target_key = target_key
        self.ignore_index = ignore_index
        self.mask_ignore_index = mask_ignore_index
        if sampler_type == "ptv3":
            from .learned import SamplerPTV3

            self.sampler = SamplerPTV3(
                sampler_backbone,
                out_channels=sampler_backbone_out_channels,
                head_hidden_channels=sampler_hidden_channels,
                head_num_layers=sampler_num_layers,
                fusion_mode=sampler_fusion_mode,
            )
        elif sampler_type == "generic":
            self.sampler = SamplerGenericBackbone(
                sampler_backbone,
                sampler_backbone_out_channels,
                sampler_hidden_channels,
                sampler_num_layers,
            )
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")

    def predict_logits(self, input_dict):
        return self.sampler(input_dict)

    def predict_proba(self, input_dict):
        return torch.sigmoid(self.predict_logits(input_dict))

    def forward(self, input_dict):
        predicted = self.predict_proba(input_dict)
        target = input_dict[self.target_key].to(predicted.device, dtype=torch.float32)
        valid = torch.isfinite(target)
        if self.mask_ignore_index and "segment" in input_dict:
            valid = valid & (input_dict["segment"] != self.ignore_index)
        loss = marginal_probability_mse_loss(predicted, target, valid_mask=valid)
        return dict(loss=loss, mse_loss=loss)
