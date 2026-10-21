# Source config: selector-ptv3m1-offline-fixed-subset-preaug-alignold-limit02-lr012-e1600.py
# eval_epoch is the total number of validation cycles.
weight = None
resume = False
evaluate = True
test_only = False
seed = 0
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
    {
        "type": "SemSegEvaluator",
        "write_cls_iou": True,
        "write_confusion": True,
        "save_metric": "subset_sparse",
    },
    {"type": "CheckpointSaver", "save_freq": 1},
]
train = {"type": "DefaultTrainer"}
test = {"type": "SemSegTester", "verbose": True}
batch_size = 12
num_worker = 8
gradient_accumulation_steps = 1
batch_size_val = 1
batch_size_test = 12
epoch = 1600
eval_epoch = 10
mix_prob = 0.8
enable_amp = True
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
param_dicts = [{"keyword": "block", "lr": 0.0012}]
optimizer = {"type": "AdamW", "lr": 0.012, "weight_decay": 0.05}
scheduler = {
    "type": "OneCycleLR",
    "max_lr": [0.012, 0.0012],
    "pct_start": 0.05,
    "anneal_strategy": "cos",
    "div_factor": 10.0,
    "final_div_factor": 1000.0,
}
data = {
    "num_classes": 13,
    "ignore_index": -1,
    "names": [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    "train": {
        "type": "S3DISDataset",
        "split": ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        "data_root": "data/s3dis",
        "transform": [
            {
                "type": "ApplySampledIndex",
                "index_key": "sampled_index",
                "reset_sampled_index": True,
                "mark_fixed_oracle_subset": True,
                "skip_train_gridsample": False,
                "preserve_sampled_index_on_point_drop": False,
            },
            {"type": "CenterShift", "apply_z": True},
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
                "keys": ("coord", "grid_coord", "segment", "sampled_index"),
                "offset_keys_dict": {
                    "offset": "coord",
                    "sampled_index_offset": "sampled_index",
                },
                "feat_keys": ("color", "normal"),
            },
        ],
        "test_mode": False,
        "lr_file": "data/s3dis/limit0.2.txt",
    },
    "val": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
        "transform": [
            {"type": "CenterShift", "apply_z": True},
            {"type": "CenterShift", "apply_z": False},
            {"type": "NormalizeColor"},
            {"type": "Update", "keys_dict": {"grid_size": 0.02}},
            {"type": "ToTensor"},
            {
                "type": "Collect",
                "keys": ("coord", "grid_size", "segment", "sampled_index"),
                "offset_keys_dict": {
                    "offset": "coord",
                    "sampled_index_offset": "sampled_index",
                },
                "feat_keys": ("color", "normal"),
            },
        ],
        "test_mode": False,
    },
    "test": {
        "type": "S3DISDataset",
        "split": "Area_5",
        "data_root": "data/s3dis",
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
    "type": "SelectorSegmentor-v1m1",
    "ignore_index": -1,
    "eval_map_to_full": True,
    "segmentor": {
        "type": "DefaultSegmentorV2",
        "num_classes": 13,
        "backbone_out_channels": 64,
        "backbone": {
            "type": "PT-v3m1",
            "in_channels": 6,
            "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
            "stride": (2, 2, 2, 2),
            "enc_depths": (2, 2, 2, 6, 2),
            "enc_channels": (32, 64, 128, 256, 512),
            "enc_num_head": (2, 4, 8, 16, 32),
            "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
            "dec_depths": (2, 2, 2, 2),
            "dec_channels": (64, 64, 128, 256),
            "dec_num_head": (4, 4, 8, 16),
            "dec_patch_size": (1024, 1024, 1024, 1024),
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "drop_path": 0.3,
            "shuffle_orders": True,
            "pre_norm": True,
            "enable_rpe": False,
            "enable_flash": True,
            "upcast_attention": False,
            "upcast_softmax": False,
            "enc_mode": False,
            "pdnorm_bn": False,
            "pdnorm_ln": False,
            "pdnorm_decouple": True,
            "pdnorm_adaptive": False,
            "pdnorm_affine": True,
            "pdnorm_conditions": ("ScanNet", "S3DIS", "Structured3D"),
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
    },
    "selector": {"type": "FixedSelector", "num_classes": 13, "ignore_index": -1},
}
