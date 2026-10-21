_base_ = ["../scannet/oracle.py"]
save_path = "outputs/ablations/segmentor_spunet_base_oracle"
model = {
    "segmentor": {
        "type": "DefaultSegmentorV2",
        "num_classes": 20,
        "backbone_out_channels": 128,
        "backbone": {
            "type": "SpUNet-v1m1",
            "in_channels": 6,
            "num_classes": 128,
            "base_channels": 32,
            "channels": (32, 64, 128, 256, 256, 128, 96, 128),
            "layers": (2, 3, 4, 6, 2, 2, 2, 2),
            "enc_mode": False,
        },
        "criteria": [
            {"type": "CrossEntropyLoss", "loss_weight": 1.0, "ignore_index": -1},
            {
                "type": "LovaszLoss",
                "mode": "multiclass",
                "loss_weight": 1.0,
                "ignore_index": -1,
            },
        ],
        "_delete_": True,
    }
}
