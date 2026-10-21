_base_ = ["../scannet/sampler.py"]
save_path = "outputs/ablations/sampler_spunet_base"
model = {
    "type": "MarginalSamplerRegressor",
    "target_key": "marginal_prob",
    "ignore_index": -1,
    "mask_ignore_index": False,
    "sampler_backbone": {
        "type": "SpUNet-v1m1",
        "in_channels": 6,
        "num_classes": 128,
        "base_channels": 32,
        "channels": (32, 64, 128, 256, 256, 128, 96, 128),
        "layers": (2, 3, 4, 6, 2, 2, 2, 2),
        "enc_mode": False,
    },
    "sampler_backbone_out_channels": 128,
    "sampler_fusion_mode": "concat",
    "sampler_hidden_channels": 128,
    "sampler_num_layers": 2,
    "sampler_type": "generic",
    "_delete_": True,
}
