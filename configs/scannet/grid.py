# Paper Grid baseline: online GridSample(0.04) after augmentation.
# Validation retains the original full cloud before voxel sampling.
weight = None

resume = False

evaluate = True

test_only = False

seed = 433766

save_path = 'outputs/scannet/grid'

clip_grad = None

sync_bn = False

enable_json_log = False

json_log_file = 'metrics.jsonl'

json_log_window_size = 20

find_unused_parameters = False

hooks = [{'type': 'CheckpointLoader'},
 {'type': 'ModelHook'},
 {'type': 'IterationTimer', 'warmup_iter': 2},
 {'type': 'InformationWriter'},
 {'type': 'SemSegEvaluator', 'write_cls_iou': True, 'write_confusion': True},
 {'type': 'CheckpointSaver', 'save_freq': 1}]

train = {'type': 'DefaultTrainer'}

test = {'type': 'SemSegTester', 'verbose': True}

batch_size = 24

num_worker = 16

gradient_accumulation_steps = 1

batch_size_val = None

batch_size_test = None

epoch = 800

eval_epoch = 10

mix_prob = 0.8

enable_amp = True

amp_dtype = 'float16'

empty_cache = False

empty_cache_per_epoch = False

param_dicts = [{'keyword': 'block', 'lr': 0.0012}]

optimizer = {'type': 'AdamW', 'lr': 0.012, 'weight_decay': 0.05}

scheduler = {'type': 'OneCycleLR',
 'max_lr': [0.012, 0.0012],
 'pct_start': 0.05,
 'anneal_strategy': 'cos',
 'div_factor': 10.0,
 'final_div_factor': 1000.0}

class_names = ['wall',
 'floor',
 'cabinet',
 'bed',
 'chair',
 'sofa',
 'table',
 'door',
 'window',
 'bookshelf',
 'picture',
 'counter',
 'desk',
 'curtain',
 'refridgerator',
 'shower curtain',
 'toilet',
 'sink',
 'bathtub',
 'otherfurniture']

dataset_type = 'ScanNetDataset'

data_root = 'data/scannet'

grid_size = 0.04

data = {'num_classes': 20,
 'ignore_index': -1,
 'names': ['wall',
           'floor',
           'cabinet',
           'bed',
           'chair',
           'sofa',
           'table',
           'door',
           'window',
           'bookshelf',
           'picture',
           'counter',
           'desk',
           'curtain',
           'refridgerator',
           'shower curtain',
           'toilet',
           'sink',
           'bathtub',
           'otherfurniture'],
 'train': {'type': 'ScanNetDataset',
           'split': 'train',
           'data_root': 'data/scannet',
           'transform': [{'type': 'CenterShift', 'apply_z': True},
                         {'type': 'RandomDropout',
                          'dropout_ratio': 0.2,
                          'dropout_application_ratio': 0.2},
                         {'type': 'RandomRotate',
                          'angle': [-1, 1],
                          'axis': 'z',
                          'center': [0, 0, 0],
                          'p': 0.5},
                         {'type': 'RandomRotate',
                          'angle': [-0.015625, 0.015625],
                          'axis': 'x',
                          'p': 0.5},
                         {'type': 'RandomRotate',
                          'angle': [-0.015625, 0.015625],
                          'axis': 'y',
                          'p': 0.5},
                         {'type': 'RandomScale', 'scale': [0.9, 1.1]},
                         {'type': 'RandomFlip', 'p': 0.5},
                         {'type': 'RandomJitter', 'sigma': 0.005, 'clip': 0.02},
                         {'type': 'ElasticDistortion',
                          'distortion_params': [[0.2, 0.4], [0.8, 1.6]]},
                         {'type': 'ChromaticAutoContrast',
                          'p': 0.2,
                          'blend_factor': None},
                         {'type': 'ChromaticTranslation', 'p': 0.95, 'ratio': 0.05},
                         {'type': 'ChromaticJitter', 'p': 0.95, 'std': 0.05},
                         {'type': 'GridSample',
                          'grid_size': 0.04,
                          'hash_type': 'fnv',
                          'mode': 'train',
                          'return_grid_coord': True},
                         {'type': 'SphereCrop', 'point_max': 102400, 'mode': 'random'},
                         {'type': 'CenterShift', 'apply_z': False},
                         {'type': 'NormalizeColor'},
                         {'type': 'ToTensor'},
                         {'type': 'Collect',
                          'keys': ('coord', 'grid_coord', 'segment'),
                          'feat_keys': ('color', 'normal')}],
           'test_mode': False,
           'lr_file': 'configs/scannet/limit0.1.txt'},
 'val': {'type': 'ScanNetDataset',
         'split': 'val',
         'data_root': 'data/scannet',
         'transform': [{'type': 'CenterShift', 'apply_z': True},
                       {'type': 'Copy',
                        'keys_dict': {'segment': 'origin_segment',
                                      'coord': 'origin_coord'}},
                       {'type': 'GridSample',
                        'grid_size': 0.04,
                        'hash_type': 'fnv',
                        'mode': 'train',
                        'return_grid_coord': True,
                        'return_inverse': True},
                       {'type': 'CenterShift', 'apply_z': False},
                       {'type': 'NormalizeColor'},
                       {'type': 'ToTensor'},
                       {'type': 'Collect',
                        'keys': ('coord',
                                 'grid_coord',
                                 'segment',
                                 'origin_segment',
                                 'origin_coord',
                                 'inverse'),
                        'feat_keys': ('color', 'normal')}],
         'test_mode': False},
 'test': {'type': 'ScanNetDataset',
          'split': 'val',
          'data_root': 'data/scannet',
          'transform': [{'type': 'CenterShift', 'apply_z': True},
                        {'type': 'NormalizeColor'}],
          'test_mode': True,
          'test_cfg': {'voxelize': {'type': 'GridSample',
                                    'grid_size': 0.04,
                                    'hash_type': 'fnv',
                                    'mode': 'test',
                                    'return_grid_coord': True},
                       'crop': None,
                       'post_transform': [{'type': 'CenterShift', 'apply_z': False},
                                          {'type': 'ToTensor'},
                                          {'type': 'Collect',
                                           'keys': ('coord', 'grid_coord', 'index'),
                                           'feat_keys': ('color', 'normal')}]}}}

model = {'type': 'DefaultSegmentorV2',
 'num_classes': 20,
 'backbone_out_channels': 64,
 'backbone': {'type': 'PT-v3m1',
              'in_channels': 6,
              'order': ('z', 'z-trans', 'hilbert', 'hilbert-trans'),
              'stride': (2, 2, 2, 2),
              'enc_depths': (2, 2, 2, 6, 2),
              'enc_channels': (32, 64, 128, 256, 512),
              'enc_num_head': (2, 4, 8, 16, 32),
              'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
              'dec_depths': (2, 2, 2, 2),
              'dec_channels': (64, 64, 128, 256),
              'dec_num_head': (4, 4, 8, 16),
              'dec_patch_size': (1024, 1024, 1024, 1024),
              'mlp_ratio': 4,
              'qkv_bias': True,
              'qk_scale': None,
              'attn_drop': 0.0,
              'proj_drop': 0.0,
              'drop_path': 0.3,
              'shuffle_orders': True,
              'pre_norm': True,
              'enable_rpe': False,
              'enable_flash': True,
              'upcast_attention': False,
              'upcast_softmax': False,
              'enc_mode': False,
              'pdnorm_bn': False,
              'pdnorm_ln': False,
              'pdnorm_decouple': True,
              'pdnorm_adaptive': False,
              'pdnorm_affine': True,
              'pdnorm_conditions': ('ScanNet', 'S3DIS', 'Structured3D')},
 'criteria': [{'type': 'CrossEntropyLoss', 'loss_weight': 1.0, 'ignore_index': -1},
              {'type': 'LovaszLoss',
               'mode': 'multiclass',
               'loss_weight': 1.0,
               'ignore_index': -1}]}
