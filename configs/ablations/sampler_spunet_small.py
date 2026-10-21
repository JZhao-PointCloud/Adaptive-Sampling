_base_ = ["../scannet/sampler.py"]
save_path = "outputs/ablations/sampler_spunet_small"
model = {
    "type": "MarginalSamplerRegressor",
    "target_key": "marginal_prob",
    "ignore_index": -1,
    "mask_ignore_index": False,
    "sampler_backbone": {
        "type": "SpUNet-v1m1",
        "in_channels": 6,
        "num_classes": 96,
        "base_channels": 16,
        "channels": (16, 32, 64, 128, 128, 64, 48, 96),
        "layers": (1, 2, 2, 3, 1, 1, 1, 1),
        "enc_mode": False,
    },
    "sampler_backbone_out_channels": 96,
    "sampler_fusion_mode": "concat",
    "sampler_hidden_channels": 128,
    "sampler_num_layers": 2,
    "sampler_type": "generic",
    "_delete_": True,
}
