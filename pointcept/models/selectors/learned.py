import torch
import torch.nn as nn
from pointcept.models.builder import build_model


class SamplerMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        assert num_layers >= 1
        layers = []
        c_in = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(c_in, hidden_channels))
            layers.append(nn.GELU())
            c_in = hidden_channels
        layers.append(nn.Linear(c_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SamplerPTV3(nn.Module):
    def __init__(
        self,
        backbone,
        out_channels=None,
        head_hidden_channels=64,
        head_num_layers=2,
        fusion_mode="concat",
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("sampler_backbone is required when sampler_type='ptv3'")
        backbone = dict(backbone)
        backbone.setdefault("enc_mode", True)
        if not backbone["enc_mode"]:
            raise ValueError(
                "sampler_backbone.enc_mode must be True for sampler_type='ptv3'"
            )
        enc_channels = backbone.get("enc_channels")
        if fusion_mode != "concat" and enc_channels is None:
            raise KeyError(
                "sampler_backbone.enc_channels is required for non-concat fusion"
            )
        if out_channels is None:
            if enc_channels is None:
                raise KeyError(
                    "sampler_backbone.enc_channels is required to infer sampler_backbone_out_channels"
                )
            out_channels = int(sum(enc_channels))
        if fusion_mode not in {"concat", "gated"}:
            raise ValueError(f"Unsupported sampler fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.backbone = build_model(backbone)
        if self.fusion_mode == "gated":
            fusion_gates = []
            child_channels = int(enc_channels[-1])
            for parent_channels in reversed(enc_channels[:-1]):
                parent_channels = int(parent_channels)
                fusion_gates.append(
                    nn.Linear(parent_channels + child_channels, child_channels)
                )
                child_channels += parent_channels
            self.fusion_gates = nn.ModuleList(fusion_gates)
        else:
            self.fusion_gates = None
        self.head = SamplerMLP(
            in_channels=out_channels,
            hidden_channels=head_hidden_channels,
            num_layers=head_num_layers,
        )

    @staticmethod
    def _prepare_point_input(input_dict):
        point_input = dict(input_dict)
        if "grid_size" in point_input and torch.is_tensor(point_input["grid_size"]):
            if point_input["grid_size"].numel() > 1:
                point_input["grid_size"] = point_input["grid_size"].reshape(-1)[0]
        return point_input

    def forward(self, input_dict, return_feat=False):
        from pointcept.models.utils.structure import Point

        point = Point(self._prepare_point_input(input_dict))
        point = self.backbone(point)
        fusion_idx = 0
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            child_feat = point.feat[inverse]
            if self.fusion_mode == "gated":
                gate_input = torch.cat([parent.feat, child_feat], dim=-1)
                child_feat = child_feat * torch.sigmoid(
                    self.fusion_gates[fusion_idx](gate_input)
                )
                fusion_idx += 1
            parent.feat = torch.cat([parent.feat, child_feat], dim=-1)
            point = parent
        score = self.head(point.feat)
        if return_feat:
            return score, point.feat
        return score
