# Source config: marginal-sampler-ptv3wide-lr3-loop10-waterfill-ignorecls-mse-limit01.py
# eval_epoch is the total number of validation cycles.
weight = None
resume = False
evaluate = True
test_only = False
seed = 433774
save_path = "exp/default"
clip_grad = None
sync_bn = False
enable_json_log = False
json_log_file = "metrics.jsonl"
json_log_window_size = 20
find_unused_parameters = False
hooks = [
    {"type": "CheckpointLoader"},
    {"type": "ModelHook"},
    {"type": "IterationTimer", "warmup_iter": 2},
    {"type": "InformationWriter"},
    {"type": "LossEvaluator"},
    {"type": "CheckpointSaver", "save_freq": 1},
]
train = {"type": "DefaultTrainer"}
test = {"type": "SemSegTester", "verbose": True}
data = {
    "num_classes": 20,
    "ignore_index": -1,
    "names": [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    "train": {
        "type": "ScanNetDataset",
        "split": "train",
        "data_root": "data/scannet",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {
                "type": "Update",
                "keys_dict": {
                    "index_valid_keys": [
                        "coord",
                        "color",
                        "normal",
                        "superpoint",
                        "strength",
                        "segment",
                        "instance",
                        "marginal_prob",
                    ]
                },
            },
            {
                "type": "RandomDropout",
                "dropout_ratio": 0.2,
                "dropout_application_ratio": 0.2,
            },
            {
                "type": "RandomRotate",
                "angle": [-1, 1],
                "axis": "z",
                "center": [0, 0, 0],
                "p": 0.5,
            },
            {
                "type": "RandomRotate",
                "angle": [-0.015625, 0.015625],
                "axis": "x",
                "p": 0.5,
            },
            {
                "type": "RandomRotate",
                "angle": [-0.015625, 0.015625],
                "axis": "y",
                "p": 0.5,
            },
            {"type": "RandomScale", "scale": [0.9, 1.1]},
            {"type": "RandomFlip", "p": 0.5},
            {"type": "RandomJitter", "sigma": 0.005, "clip": 0.02},
            {
                "type": "ElasticDistortion",
                "distortion_params": [[0.2, 0.4], [0.8, 1.6]],
            },
            {"type": "ChromaticAutoContrast", "p": 0.2, "blend_factor": None},
            {"type": "ChromaticTranslation", "p": 0.95, "ratio": 0.05},
            {"type": "ChromaticJitter", "p": 0.95, "std": 0.05},
            {
                "type": "GridSample",
                "grid_size": 0.02,
                "hash_type": "fnv",
                "mode": "train",
                "return_grid_coord": True,
            },
            {"type": "SphereCrop", "point_max": 102400, "mode": "random"},
            {"type": "CenterShift", "apply_z": False},
            {"type": "NormalizeColor"},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_coord", "segment", "marginal_prob"),
                "feat_keys": ("color", "normal"),
            },
        ],
        "test_mode": False,
        "lr_file": "configs/scannet/limit0.1.txt",
        "marginal_prob_file": "artifacts/scannet/oracle/train_probabilities.pth",
    },
    "val": {
        "type": "ScanNetDataset",
        "split": "val",
        "data_root": "data/scannet",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {"type": "NormalizeColor"},
            {"type": "Update", "keys_dict": {"grid_size": 0.02}},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_size", "segment", "marginal_prob"),
                "feat_keys": ("color", "normal"),
            },
        ],
        "test_mode": False,
        "marginal_prob_file": "artifacts/scannet/oracle/val_probabilities.pth",
    },
    "test": {
        "type": "ScanNetDataset",
        "split": "val",
        "data_root": "data/scannet",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {"type": "NormalizeColor"},
            {"type": "Update", "keys_dict": {"grid_size": 0.02}},
        ],
        "test_mode": True,
        "test_cfg": {
            "voxelize": None,
            "crop": None,
            "post_transform": [
                {"type": "CenterShift", "apply_z": False},
                {"type": "ToTensor"},
                {
                    "type": "Collect",
                    "keys": ("coord", "grid_size", "index"),
                    "feat_keys": ("color", "normal"),
                },
            ],
            "aug_transform": [
                [
                    {
                        "type": "RandomRotateTargetAngle",
                        "angle": [0],
                        "axis": "z",
                        "center": [0, 0, 0],
                        "p": 1,
                    }
                ]
            ],
        },
    },
}
model = {
    "type": "MarginalSamplerRegressor",
    "target_key": "marginal_prob",
    "ignore_index": -1,
    "mask_ignore_index": False,
    "sampler_backbone": {
        "type": "PT-v3m1",
        "in_channels": 6,
        "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
        "stride": (2, 2),
        "enc_depths": (2, 4, 6),
        "enc_channels": (24, 48, 96),
        "enc_num_head": (2, 4, 8),
        "enc_patch_size": (256, 256, 256),
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.1,
        "shuffle_orders": True,
        "pre_norm": True,
        "enable_rpe": False,
        "enable_flash": True,
        "upcast_attention": False,
        "upcast_softmax": False,
        "enc_mode": True,
    },
    "sampler_backbone_out_channels": 168,
    "sampler_fusion_mode": "concat",
    "sampler_hidden_channels": 128,
    "sampler_num_layers": 2,
}
batch_size = 12
batch_size_val = 12
batch_size_test = None
num_worker = 8
gradient_accumulation_steps = 1
epoch = 2000
eval_epoch = 5
mix_prob = 0.0
enable_amp = True
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
param_dicts = None
optimizer = {"type": "AdamW", "lr": 0.003, "weight_decay": 0.05}
scheduler = {
    "type": "OneCycleLR",
    "max_lr": 0.003,
    "pct_start": 0.05,
    "anneal_strategy": "cos",
    "div_factor": 10.0,
    "final_div_factor": 1000.0,
}

oracle_include_ignore_as_class = True
