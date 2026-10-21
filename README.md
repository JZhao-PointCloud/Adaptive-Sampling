# PointSampler

Semantic-aware point sampling for 3D segmentation, built on [Pointcept](https://github.com/Pointcept/Pointcept).

Includes Grid and Oracle baselines, MSE sampler training, segmentor retraining, and multi-mask KNN evaluation on ScanNet, S3DIS and PartNet. SpUNet backbone ablations are in `configs/ablations/`.

## Installation

Use Python 3.10–3.12, Linux and an NVIDIA GPU. Install CUDA PyTorch, matching `torch-scatter` and `spconv`, then:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
pip install packaging ninja wheel
pip install flash-attn --no-build-isolation
pip install ./libs/pointops --no-build-isolation
```

## Usage

Place Pointcept-preprocessed datasets under `data/scannet`, `data/s3dis` and `data/partnet`. Run commands from the repository root.

ScanNet example (2 GPUs, using the included 120-scene training split):

```bash
# Oracle indices and sampler targets
python -m tools.prepare --config configs/scannet/oracle.py --kind oracle --output-root artifacts/scannet/oracle --base-grid-size 0.02 --target-grid-size 0.04
python -m tools.prepare --config configs/scannet/sampler.py --kind probabilities --output-root artifacts/scannet/oracle --base-grid-size 0.02 --target-grid-size 0.04

# Baselines and sampler
python -m tools.train --config-file configs/scannet/grid.py --num-gpus 2
python -m tools.train --config-file configs/scannet/oracle.py --num-gpus 2
python -m tools.train --config-file configs/scannet/sampler.py --num-gpus 2

# Learned indices and segmentor retraining
python -m tools.precompute_learned_sampler_indices --config configs/scannet/sampler_eval.py --checkpoint outputs/scannet/sampler/model/model_best.pth --output-root artifacts/scannet/learned --output-name indices --splits train val --train-transform-source val --selection-mode calibrated_bernoulli --target-grid-size 0.04 --final-grid-cleanup-size 0.02 --seed 433766 --allow-point-count-change
python -m tools.train --config-file configs/scannet/retrain.py --num-gpus 2 --options weight=outputs/scannet/oracle/model/model_best.pth

# Evaluation
python -m tools.eval_multimask_sampler_ensemble --sampler-config configs/scannet/sampler_eval.py --sampler-checkpoint outputs/scannet/sampler/model/model_best.pth --segmentor-config configs/scannet/retrain.py --segmentor-checkpoint outputs/scannet/retrain/model/model_eval8.pth --output-root outputs/scannet/multimask --masks 32 --prefix-masks 1,2,4,8,16,32 --knn-k-list 1,3 --full-score-mode logit --seed 433766
```

Results are saved as JSON and CSV under `outputs/`. Change `--masks` and `--knn-k-list` for aggregation ablations.

Subset evaluates selected points; KNN evaluates the original full point cloud. Keep the complete preprocessed clouds in `data/`; sampling indices are stored separately in `artifacts/`.

Other datasets use the same stages with configs in `configs/s3dis/` and `configs/partnet/`. Training subsets are 20% for S3DIS and 10% for PartNet. Set the PartNet sampling budget explicitly.

Oracle targets use class-then-voxel waterfill. Evaluation averages mask logits; `prob` applies softmax to that average before KNN. For post-cleanup budget matching, add `--calibrate-cleaned-count` to both index generation and evaluation.

Training schedules follow the original configs: S3DIS segmentor LR 0.012; PartNet sampler LR 0.001 for 200 loops. ScanNet uses global batches 24/12/24 and training loops 800/2000/1600 for Oracle/Sampler/retraining; the paper evaluation uses retraining checkpoint `model_eval8`. ScanNet Grid uses online 0.04 m voxel sampling, LR 0.012, global batch 24 and 800 loops. PartNet sampler batch size is 4. `eval_epoch` is the total number of validation cycles.

## Tests

With PyTorch installed:

```bash
pip install -r requirements-test.txt
pip install -e . --no-deps
python -m pytest -q
```

Pointcept's [MIT license](third_party/Pointcept-LICENSE) is included.
