_base_ = ["../scannet/oracle.py"]
save_path = "outputs/ablations/segmentor_spunet_small_oracle"
model = {
    "segmentor": {
        "type": "DefaultSegmentorV2",
        "num_classes": 20,
        "backbone_out_channels": 96,
        "backbone": {
            "type": "SpUNet-v1m1",
            "in_channels": 6,
            "num_classes": 96,
            "base_channels": 16,
            "channels": (16, 32, 64, 128, 128, 64, 48, 96),
            "layers": (1, 2, 2, 3, 1, 1, 1, 1),
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
