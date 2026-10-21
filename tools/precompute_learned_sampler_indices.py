import argparse
import copy
import json
import random
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from pointcept.datasets import build_dataset
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from tools.precompute_fixed_selector_indices import (
    apply_final_cleanup,
    deterministic_rng,
    deterministic_tie_key,
    grid_count,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute selected_index files from a trained marginal sampler"
    )
    parser.add_argument("--config", required=True, help="Sampler training config")
    parser.add_argument(
        "--checkpoint", required=True, help="Trained sampler checkpoint."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", default=None, help="Optional data_root override")
    parser.add_argument(
        "--train-transform-source",
        choices=["train", "val"],
        default="val",
        help="Transform to use for split=train. Use val when the train transform contains stochastic point-dropping augmentations but output indices must still refer to raw train scenes.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["bernoulli", "topk", "calibrated_bernoulli"],
        default="calibrated_bernoulli",
    )
    parser.add_argument("--target-grid-size", type=float, default=0.04)
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--min-sample-points", type=int, default=256)
    parser.add_argument("--final-grid-cleanup-size", type=float, default=None)
    parser.add_argument(
        "--calibrate-cleaned-count",
        action="store_true",
        help="For calibrated_bernoulli, binary-search the bias against the final cleanup-selected count instead of the expected raw Bernoulli count.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--allow-point-count-change",
        action="store_true",
        help="Allow transforms that drop/grid points. Off by default because output indices must refer to raw scenes.",
    )
    return parser.parse_args()


def load_model(config_path, checkpoint_path=None):
    if checkpoint_path is None:
        raise ValueError("A trained sampler checkpoint is required")
    from pointcept.models import build_model

    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    weight = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    model.load_state_dict(weight, strict=True)
    model.eval()
    return (cfg, model)


def build_split_dataset(cfg, split, data_root=None, train_transform_source="train"):
    if split == "train" and train_transform_source == "val":
        data_cfg = copy.deepcopy(cfg.data.train)
        data_cfg.transform = copy.deepcopy(cfg.data.val.transform)
    elif split in cfg.data:
        data_cfg = copy.deepcopy(cfg.data[split])
    else:
        data_cfg = copy.deepcopy(cfg.data.val)
        data_cfg.split = split
    if data_root is not None:
        data_cfg.data_root = data_root
    data_cfg.pop("marginal_prob_file", None)
    data_cfg.pop("selected_index_file", None)
    for transform in data_cfg.transform:
        if transform.get("type") == "Collect":
            transform["keys"] = tuple(
                k for k in transform["keys"] if k != "marginal_prob"
            )
    data_cfg.test_mode = False
    data_cfg.loop = 1
    return build_dataset(data_cfg)


def resolve_target_count(coord, *, target_grid_size, budget_ratio, min_sample_points):
    target_count = grid_count(coord, float(target_grid_size))
    target_count = int(np.ceil(target_count * float(budget_ratio)))
    target_count = max(int(min_sample_points), target_count)
    target_count = min(int(coord.shape[0]), target_count)
    return int(target_count)


def calibrate_probability_to_target_count(logit, target_count, *, max_iter=64):
    logit = np.asarray(logit, dtype=np.float64).reshape(-1)
    num_points = int(logit.shape[0])
    target_count = max(0, min(int(target_count), num_points))
    if target_count <= 0:
        return np.zeros(num_points, dtype=np.float64)
    if target_count >= num_points:
        return np.ones(num_points, dtype=np.float64)
    finite = np.isfinite(logit)
    if not finite.all():
        logit = np.where(finite, logit, -60.0)
    lo = float(logit.min() - 60.0)
    hi = float(logit.max() + 60.0)
    for _ in range(int(max_iter)):
        bias = (lo + hi) / 2.0
        probability = 1.0 / (1.0 + np.exp(np.clip(-(logit - bias), -60.0, 60.0)))
        if float(probability.sum()) > float(target_count):
            lo = bias
        else:
            hi = bias
    bias = (lo + hi) / 2.0
    return 1.0 / (1.0 + np.exp(np.clip(-(logit - bias), -60.0, 60.0)))


def sample_predicted_probability(
    probability, *, mode, scene_key, seed, target_count=None, logit=None
):
    if mode == "calibrated_bernoulli":
        if logit is None:
            raise ValueError("logit is required for calibrated_bernoulli selection")
        if target_count is None:
            raise ValueError(
                "target_count is required for calibrated_bernoulli selection"
            )
        probability = calibrate_probability_to_target_count(logit, target_count)
    else:
        probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    probability = np.clip(probability, 0.0, 1.0)
    num_points = int(probability.shape[0])
    if mode in {"bernoulli", "calibrated_bernoulli"}:
        rng = deterministic_rng(f"learned:{seed}:{scene_key}", num_points)
        return np.nonzero(rng.random(num_points) < probability)[0].astype(np.int64)
    if mode == "topk":
        if target_count is None:
            raise ValueError("target_count is required for topk selection")
        target_count = max(0, min(int(target_count), num_points))
        tie_key = deterministic_tie_key(
            num_points, f"learned:{seed}:{scene_key}", target_count
        )
        order = np.lexsort((tie_key, -probability))
        return np.sort(order[:target_count].astype(np.int64, copy=False))
    raise ValueError(f"Unsupported selection mode: {mode}")


def sample_calibrated_bernoulli_cleaned_count(
    logit,
    coord,
    *,
    target_count,
    cleanup_grid_size,
    scene_key,
    seed,
    source_index=None,
    max_iter=64,
):
    logit = np.asarray(logit, dtype=np.float64).reshape(-1)
    num_points = int(logit.shape[0])
    if source_index is not None:
        source_index = np.asarray(source_index, dtype=np.int64).reshape(-1)
        if source_index.shape[0] != num_points:
            raise ValueError(
                f"source_index length must match logit length for cleaned-count calibration, got {source_index.shape[0]} vs {num_points}"
            )
    target_count = max(0, min(int(target_count), num_points))
    if target_count <= 0 or num_points == 0:
        return np.empty((0,), dtype=np.int64)
    if target_count >= num_points:
        selected = (
            source_index
            if source_index is not None
            else np.arange(num_points, dtype=np.int64)
        )
        return apply_final_cleanup(coord, selected, cleanup_grid_size)
    finite = np.isfinite(logit)
    if not finite.all():
        logit = np.where(finite, logit, -60.0)
    uniform = deterministic_rng(f"learned:{seed}:{scene_key}", num_points).random(
        num_points
    )

    def select_at_bias(bias):
        probability = 1.0 / (1.0 + np.exp(np.clip(-(logit - float(bias)), -60.0, 60.0)))
        selected = np.nonzero(uniform < probability)[0].astype(np.int64, copy=False)
        if source_index is not None:
            selected = source_index[selected]
        return apply_final_cleanup(coord, selected, cleanup_grid_size)

    lo = float(logit.min() - 60.0)
    hi = float(logit.max() + 60.0)
    best = select_at_bias(lo)
    best_diff = abs(int(best.shape[0]) - target_count)
    for candidate in (hi,):
        selected = select_at_bias(candidate)
        diff = abs(int(selected.shape[0]) - target_count)
        if diff < best_diff:
            best = selected
            best_diff = diff
    for _ in range(int(max_iter)):
        bias = (lo + hi) / 2.0
        selected = select_at_bias(bias)
        count = int(selected.shape[0])
        diff = abs(count - target_count)
        if diff < best_diff:
            best = selected
            best_diff = diff
            if best_diff == 0:
                break
        if count > target_count:
            lo = bias
        else:
            hi = bias
    return np.sort(np.asarray(best, dtype=np.int64))


def raw_coord_count(dataset, idx):
    data_path = Path(dataset.data_list[idx % len(dataset.data_list)])
    return int(np.load(data_path / "coord.npy", mmap_mode="r").shape[0])


def raw_coord(dataset, idx):
    data_path = Path(dataset.data_list[idx % len(dataset.data_list)])
    return np.load(data_path / "coord.npy").astype(np.float32, copy=False)


def seed_scene_transform(scene_key, seed):
    rng = deterministic_rng(f"transform:{int(seed)}:{scene_key}", 0)
    scene_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    torch.manual_seed(scene_seed)


def tensor_to_numpy_int(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).astype(np.int64, copy=False)


def resolve_source_index(data, *, transformed_count, source_count, scene_key):
    if "point_index" not in data:
        return None
    source_index = tensor_to_numpy_int(data["point_index"])
    if source_index.shape[0] != int(transformed_count):
        raise RuntimeError(
            f"{scene_key} point_index length {source_index.shape[0]} does not match transformed point count {transformed_count}"
        )
    if source_index.size == 0:
        return source_index
    if source_index.min() < 0 or source_index.max() >= int(source_count):
        raise RuntimeError(
            f"{scene_key} point_index is out of raw-scene range [0, {source_count})"
        )
    return source_index


def prepare_model_tensor(value):
    return value.contiguous()


def infer_scene_scores(model, data):
    input_dict = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = prepare_model_tensor(value).cuda(non_blocking=True)
        else:
            input_dict[key] = value
    with torch.no_grad():
        if hasattr(model, "predict_logits"):
            logit = model.predict_logits(input_dict)
            probability = torch.sigmoid(logit)
        else:
            probability = model.predict_proba(input_dict)
            logit = torch.logit(probability.clamp(1e-06, 1.0 - 1e-06))
    return (
        probability.detach().float().cpu().numpy(),
        logit.detach().float().cpu().numpy(),
    )


def output_filename(split, args):
    if args.output_name is not None:
        stem = (
            args.output_name[:-4]
            if args.output_name.endswith(".pth")
            else args.output_name
        )
        return f"{split}_{stem}.pth"
    parts = ["learned_marginal", args.selection_mode]
    if args.selection_mode in {"topk", "calibrated_bernoulli"}:
        parts.append(f"aimg{str(args.target_grid_size).replace('.', '')}")
    if args.final_grid_cleanup_size is not None:
        parts.append(f"cleang{str(args.final_grid_cleanup_size).replace('.', '')}")
    if args.selection_mode == "calibrated_bernoulli" and args.calibrate_cleaned_count:
        parts.append("cleanmatch")
    return f"{split}_{'_'.join(parts)}.pth"


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg, model = load_model(args.config, args.checkpoint)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        dataset = build_split_dataset(
            cfg,
            split,
            data_root=args.data_root,
            train_transform_source=args.train_transform_source,
        )
        selected_index_dict = {}
        selection_stats = []
        total = len(dataset.data_list)
        for idx in range(total):
            scene_name = dataset.get_data_name(idx)
            scene_key = f"{split}:{scene_name}"
            source_count = raw_coord_count(dataset, idx)
            seed_scene_transform(scene_key, args.seed)
            data = dataset[idx]
            transformed_count = int(data["coord"].shape[0])
            source_index = resolve_source_index(
                data,
                transformed_count=transformed_count,
                source_count=source_count,
                scene_key=scene_key,
            )
            if transformed_count != source_count and (
                not args.allow_point_count_change
            ):
                raise RuntimeError(
                    f"{scene_key} transform changed point count from {source_count} to {transformed_count}; selected_index would no longer refer to raw scene"
                )
            if transformed_count != source_count and source_index is None:
                raise RuntimeError(
                    f"{scene_key} transform changed point count from {source_count} to {transformed_count}, but no point_index was collected to remap selected indices back to the raw scene"
                )
            probability, logit = infer_scene_scores(model, data)
            coord = raw_coord(dataset, idx)
            target_count = None
            if args.selection_mode in {"topk", "calibrated_bernoulli"}:
                target_count = resolve_target_count(
                    coord,
                    target_grid_size=args.target_grid_size,
                    budget_ratio=args.budget_ratio,
                    min_sample_points=args.min_sample_points,
                )
            if (
                args.selection_mode == "calibrated_bernoulli"
                and args.calibrate_cleaned_count
            ):
                if args.final_grid_cleanup_size is None:
                    raise ValueError(
                        "--calibrate-cleaned-count requires --final-grid-cleanup-size"
                    )
                selected_index = sample_calibrated_bernoulli_cleaned_count(
                    logit,
                    coord,
                    target_count=target_count,
                    cleanup_grid_size=args.final_grid_cleanup_size,
                    scene_key=scene_key,
                    seed=args.seed,
                    source_index=source_index,
                )
            else:
                selected_index = sample_predicted_probability(
                    probability,
                    mode=args.selection_mode,
                    scene_key=scene_key,
                    seed=args.seed,
                    target_count=target_count,
                    logit=logit,
                )
                if source_index is not None:
                    selected_index = source_index[selected_index]
                selected_index = apply_final_cleanup(
                    coord, selected_index, args.final_grid_cleanup_size
                )
            selected_index_dict[scene_name] = selected_index.astype(
                np.int64, copy=False
            )
            selection_stats.append(
                dict(
                    scene=scene_name, target=target_count, selected=len(selected_index)
                )
            )
            if idx == 0 or idx + 1 == total or (idx + 1) % 10 == 0:
                print(
                    f"[{split}] done={idx + 1}/{total} scene={scene_name} target={target_count} selected={selected_index.shape[0]}",
                    flush=True,
                )
        output_path = output_root / output_filename(split, args)
        torch.save(selected_index_dict, output_path)
        output_path.with_suffix(".json").write_text(
            json.dumps(dict(args=vars(args), scenes=selection_stats), indent=2)
        )
        print(output_path, flush=True)


if __name__ == "__main__":
    main()
