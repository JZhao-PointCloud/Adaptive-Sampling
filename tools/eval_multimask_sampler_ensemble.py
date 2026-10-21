import argparse
import copy
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from pointcept.datasets import build_dataset
from pointcept.utils.config import Config
from pointcept.utils.env import set_seed
from tools.precompute_fixed_selector_indices import apply_final_cleanup
from tools.precompute_learned_sampler_indices import (
    build_split_dataset,
    infer_scene_scores,
    load_model as load_sampler_model,
    raw_coord,
    resolve_source_index,
    resolve_target_count,
    sample_calibrated_bernoulli_cleaned_count,
    sample_predicted_probability,
    seed_scene_transform,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed downstream selector-segmentor with a test-time ensemble of multiple learned sampler masks."
    )
    parser.add_argument("--sampler-config", required=True)
    parser.add_argument("--sampler-checkpoint", required=True)
    parser.add_argument("--segmentor-config", required=True)
    parser.add_argument("--segmentor-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument(
        "--prefix-masks",
        default=None,
        help="Comma-separated mask prefixes to evaluate from a single multi-mask run, for example 1,2,4,8,16,24,32. Defaults to --masks.",
    )
    parser.add_argument("--seed", type=int, default=433766)
    parser.add_argument("--target-grid-size", type=float, default=0.04)
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--min-sample-points", type=int, default=256)
    parser.add_argument("--final-grid-cleanup-size", type=float, default=0.02)
    parser.add_argument(
        "--calibrate-cleaned-count",
        action="store_true",
        help="Binary-search each Bernoulli mask against post-cleanup count. This is more budget-faithful but much slower.",
    )
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument(
        "--knn-k-list",
        default=None,
        help="Comma-separated full-lift K values to evaluate in one pass. Defaults to --knn-k for backward compatibility.",
    )
    parser.add_argument(
        "--full-score-mode",
        default="logit",
        help="KNN input scores: logit, prob, or both. Mask logits are averaged first; 'prob' then applies softmax before KNN interpolation.",
    )
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    parser.add_argument("--scene-shard-index", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--print-freq", type=int, default=10)
    return parser.parse_args()


def parse_int_list(value, *, default):
    if value is None or str(value).strip() == "":
        values = [int(default)]
    else:
        values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one K value is required")
    result = []
    for value in values:
        if value < 1:
            raise ValueError(f"K must be >= 1, got {value}")
        if value not in result:
            result.append(value)
    return result


def parse_score_modes(value):
    modes = []
    for part in str(value).split(","):
        mode = part.strip().lower()
        if not mode:
            continue
        if mode in {"both", "all"}:
            candidates = ["logit", "prob"]
        elif mode in {"logit", "prob"}:
            candidates = [mode]
        else:
            raise ValueError(f"Unsupported --full-score-mode: {mode}")
        for candidate in candidates:
            if candidate not in modes:
                modes.append(candidate)
    if not modes:
        raise ValueError("At least one full score mode is required")
    return modes


def full_metric_name(k, score_mode):
    suffix = "" if score_mode == "logit" else f"_{score_mode}"
    return f"full_knn{int(k)}_idw{suffix}"


def new_confusions(num_classes, metric_names):
    return {
        name: np.zeros((num_classes, num_classes), dtype=np.int64)
        for name in metric_names
    }


def strip_module_state_dict(state_dict):
    weight = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    return weight


def load_segmentor(config_path, checkpoint_path):
    from pointcept.models import build_model

    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(strip_module_state_dict(state_dict), strict=True)
    model.eval()
    if hasattr(model, "eval_map_to_full"):
        model.eval_map_to_full = False
    return (cfg, model)


def make_dummy_selected_index_file(data_cfg, output_root, split):
    data_root = Path(data_cfg.data_root)
    split_name = getattr(data_cfg, "split", split)
    if isinstance(split_name, (list, tuple)):
        split_names = list(split_name)
    else:
        split_names = [split_name]
    selected = {}
    for current_split in split_names:
        split_root = data_root / current_split
        for scene_dir in sorted((p for p in split_root.iterdir() if p.is_dir())):
            selected[scene_dir.name] = np.array([0], dtype=np.int64)
    if not selected:
        raise RuntimeError(
            f"No scenes found under data_root={data_root} split={split_name}"
        )
    path = output_root / f"dummy_{split}_selected_index.pth"
    torch.save(selected, path)
    return str(path)


def build_segmentor_dataset(cfg, split, output_root):
    if split in cfg.data:
        data_cfg = copy.deepcopy(cfg.data[split])
    else:
        data_cfg = copy.deepcopy(cfg.data.val)
        data_cfg.split = split
    data_cfg.test_mode = False
    data_cfg.loop = 1
    data_cfg.selected_index_file = make_dummy_selected_index_file(
        data_cfg, output_root, split
    )
    return build_dataset(data_cfg)


def to_cuda_input(input_dict, selected_index):
    output = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.cuda(non_blocking=True)
        else:
            output[key] = value
    selected = torch.as_tensor(selected_index, dtype=torch.long, device="cuda")
    output["sampled_index"] = selected
    output["sampled_index_offset"] = torch.tensor(
        [selected.numel()], dtype=torch.long, device="cuda"
    )
    return output


def scene_key_for_mask(split, scene_name, mask_idx):
    if mask_idx == 0:
        return f"{split}:{scene_name}"
    return f"{split}:{scene_name}:m{mask_idx}"


def generate_masks(args, scene_name, coord, logit, target_count, source_index=None):
    masks = []
    for mask_idx in range(args.masks):
        scene_key = scene_key_for_mask(args.split, scene_name, mask_idx)
        seed = int(args.seed) + int(mask_idx)
        if args.calibrate_cleaned_count:
            selected = sample_calibrated_bernoulli_cleaned_count(
                logit,
                coord,
                target_count=target_count,
                cleanup_grid_size=args.final_grid_cleanup_size,
                scene_key=scene_key,
                seed=seed,
                source_index=source_index,
            )
        else:
            selected = sample_predicted_probability(
                None,
                mode="calibrated_bernoulli",
                scene_key=scene_key,
                seed=seed,
                target_count=target_count,
                logit=logit,
            )
            if source_index is not None:
                selected = np.asarray(source_index, dtype=np.int64).reshape(-1)[
                    selected
                ]
            selected = apply_final_cleanup(
                coord, selected, args.final_grid_cleanup_size
            )
        masks.append(np.asarray(selected, dtype=np.int64).reshape(-1))
    return masks


def confusion_from_pred(pred, segment, num_classes, ignore_index):
    valid = segment != int(ignore_index)
    if not torch.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    encoded = segment[valid].long() * int(num_classes) + pred[valid].long()
    confusion = torch.bincount(
        encoded, minlength=int(num_classes) * int(num_classes)
    ).reshape(int(num_classes), int(num_classes))
    return confusion.detach().cpu().numpy().astype(np.int64)


def metric_from_confusion(confusion):
    confusion = confusion.astype(np.float64)
    intersection = np.diag(confusion)
    target = confusion.sum(axis=1)
    pred = confusion.sum(axis=0)
    union = target + pred - intersection
    iou_class = intersection / (union + 1e-10)
    acc_class = intersection / (target + 1e-10)
    return dict(
        mIoU=float(np.mean(iou_class)),
        mAcc=float(np.mean(acc_class)),
        allAcc=float(intersection.sum() / (target.sum() + 1e-10)),
        iou_class=[float(x) for x in iou_class],
        acc_class=[float(x) for x in acc_class],
        target=[float(x) for x in target],
    )


def interpolate_scores_knn(sparse_scores, sparse_coord, full_coord, k):
    import pointops

    if sparse_scores.numel() == 0:
        return sparse_scores.new_zeros((full_coord.shape[0], sparse_scores.shape[-1]))
    sparse_scores = sparse_scores.contiguous()
    sparse_coord = sparse_coord.contiguous()
    full_coord = full_coord.contiguous()
    k = max(1, min(int(k), int(sparse_coord.shape[0])))
    sparse_offset = torch.tensor(
        [sparse_coord.shape[0]], device=sparse_coord.device, dtype=torch.int32
    )
    full_offset = torch.tensor(
        [full_coord.shape[0]], device=full_coord.device, dtype=torch.int32
    )
    index, distance = pointops.knn_query(
        k, sparse_coord.float(), sparse_offset, full_coord.float(), full_offset
    )
    index = index.long().reshape(-1, k)
    if k == 1:
        return sparse_scores[index.reshape(-1)]
    distance = distance.reshape(-1, k).to(sparse_scores.dtype)
    inv_distance = 1.0 / distance.clamp_min(1e-08)
    exact_mask = distance <= 1e-08
    if exact_mask.any():
        exact_weight = exact_mask.to(sparse_scores.dtype)
        has_exact = exact_mask.any(dim=1, keepdim=True)
        weights = torch.where(has_exact, exact_weight, inv_distance)
    else:
        weights = inv_distance
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-08)
    gathered = sparse_scores[index]
    return (gathered * weights.unsqueeze(-1)).sum(dim=1)


def interpolate_logits_knn(
    sparse_logits, sparse_coord, full_coord, k, score_mode="logit"
):
    if score_mode == "logit":
        sparse_scores = sparse_logits
    elif score_mode == "prob":
        sparse_scores = torch.softmax(sparse_logits, dim=-1)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    return interpolate_scores_knn(sparse_scores, sparse_coord, full_coord, k)


def add_metric(confusions, name, logits, segment, num_classes, ignore_index):
    pred = logits.max(1)[1]
    confusions[name] += confusion_from_pred(pred, segment, num_classes, ignore_index)


def add_union_metrics(
    confusions,
    union_logits,
    union_coord,
    union_segment,
    full_coord,
    full_segment,
    num_classes,
    ignore_index,
    knn_k_list,
    full_score_modes,
):
    add_metric(
        confusions,
        "subset_sparse",
        union_logits,
        union_segment,
        num_classes,
        ignore_index,
    )
    full_logits_k1 = interpolate_logits_knn(union_logits, union_coord, full_coord, 1)
    add_metric(
        confusions, "main", full_logits_k1, full_segment, num_classes, ignore_index
    )
    for k in knn_k_list:
        for score_mode in full_score_modes:
            full_scores_knn = interpolate_logits_knn(
                union_logits, union_coord, full_coord, int(k), score_mode=score_mode
            )
            add_metric(
                confusions,
                full_metric_name(k, score_mode),
                full_scores_knn,
                full_segment,
                num_classes,
                ignore_index,
            )


def scene_index(dataset):
    index = {}
    for i in range(len(dataset.data_list)):
        name = dataset.get_data_name(i)
        if name in index:
            raise ValueError(f"Duplicate scene name: {name}")
        index[name] = i
    return index


def match_scene_names(sampler_index, segmentor_index):
    missing_sampler = sorted(set(segmentor_index) - set(sampler_index))
    missing_segmentor = sorted(set(sampler_index) - set(segmentor_index))
    if missing_sampler or missing_segmentor:
        raise ValueError(
            f"Evaluation scene lists differ: missing from sampler={missing_sampler}; "
            f"missing from segmentor={missing_segmentor}"
        )
    if not segmentor_index:
        raise ValueError("Evaluation scene list is empty")
    return sorted(segmentor_index)


def main():
    args = parse_args()
    set_seed(args.seed)
    knn_k_list = parse_int_list(args.knn_k_list, default=args.knn_k)
    prefix_masks = parse_int_list(args.prefix_masks, default=args.masks)
    for prefix in prefix_masks:
        if prefix > int(args.masks):
            raise ValueError(
                f"--prefix-masks contains {prefix}, but --masks is only {args.masks}"
            )
    full_score_modes = parse_score_modes(args.full_score_mode)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = vars(args).copy()
    config_payload["resolved_knn_k_list"] = knn_k_list
    config_payload["resolved_prefix_masks"] = prefix_masks
    config_payload["resolved_full_score_modes"] = full_score_modes
    (output_root / "args.json").write_text(json.dumps(config_payload, indent=2))
    sampler_cfg, sampler_model = load_sampler_model(
        args.sampler_config, args.sampler_checkpoint
    )
    segmentor_cfg, segmentor_model = load_segmentor(
        args.segmentor_config, args.segmentor_checkpoint
    )
    sampler_dataset = build_split_dataset(sampler_cfg, args.split)
    segmentor_dataset = build_segmentor_dataset(segmentor_cfg, args.split, output_root)
    sampler_index = scene_index(sampler_dataset)
    segmentor_index = scene_index(segmentor_dataset)
    scene_names = match_scene_names(sampler_index, segmentor_index)
    if args.limit_scenes is not None:
        scene_names = scene_names[: int(args.limit_scenes)]
    shard_count = int(args.scene_shard_count)
    shard_index = int(args.scene_shard_index)
    if shard_count < 1:
        raise ValueError("--scene-shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"--scene-shard-index must be in [0, {shard_count}), got {shard_index}"
        )
    if shard_count > 1:
        scene_names = scene_names[shard_index::shard_count]
    if not scene_names:
        raise RuntimeError(
            "No overlapping scenes between sampler and segmentor datasets"
        )
    num_classes = int(segmentor_cfg.data.num_classes)
    ignore_index = int(segmentor_cfg.data.ignore_index)
    metric_names = ["main", "subset_sparse"]
    for k in knn_k_list:
        for score_mode in full_score_modes:
            metric_names.append(full_metric_name(k, score_mode))
    confusions = new_confusions(num_classes, metric_names)
    prefix_confusions = {
        int(prefix): new_confusions(num_classes, metric_names)
        for prefix in prefix_masks
    }
    scene_rows = []
    prefix_scene_rows = []
    totals = dict(raw=0, target=0, draw_total=0, union=0, valid_union=0, valid_raw=0)
    prefix_totals = {
        int(prefix): dict(
            raw=0, target=0, draw_total=0, union=0, valid_union=0, valid_raw=0
        )
        for prefix in prefix_masks
    }
    mask_totals = np.zeros(int(args.masks), dtype=np.int64)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    start_time = time.time()
    for scene_pos, scene_name in enumerate(scene_names):
        seed_scene_transform(f"{args.split}:{scene_name}", args.seed)
        sampler_data = sampler_dataset[sampler_index[scene_name]]
        probability, logit = infer_scene_scores(sampler_model, sampler_data)
        coord_raw = raw_coord(sampler_dataset, sampler_index[scene_name])
        source_index = resolve_source_index(
            sampler_data,
            transformed_count=int(logit.shape[0]),
            source_count=int(coord_raw.shape[0]),
            scene_key=f"{args.split}:{scene_name}",
        )
        if int(logit.shape[0]) != int(coord_raw.shape[0]) and source_index is None:
            raise RuntimeError(
                f"{scene_name}: sampler transform changed point count from {coord_raw.shape[0]} to {logit.shape[0]}, but no point_index was available to map selected indices back to raw scene indices"
            )
        target_count = resolve_target_count(
            coord_raw,
            target_grid_size=args.target_grid_size,
            budget_ratio=args.budget_ratio,
            min_sample_points=args.min_sample_points,
        )
        masks = generate_masks(
            args, scene_name, coord_raw, logit, target_count, source_index=source_index
        )
        segmentor_input_cpu = segmentor_dataset[segmentor_index[scene_name]]
        if not np.array_equal(
            coord_raw, raw_coord(segmentor_dataset, segmentor_index[scene_name])
        ):
            raise ValueError(
                f"{scene_name}: sampler and segmentor raw coordinates differ"
            )
        full_count = int(segmentor_input_cpu["coord"].shape[0])
        if full_count != int(coord_raw.shape[0]):
            raise RuntimeError(
                f"{scene_name}: sampler/raw count {coord_raw.shape[0]} != segmentor input count {full_count}"
            )
        full_coord = segmentor_input_cpu["coord"].cuda(non_blocking=True).float()
        full_segment = segmentor_input_cpu["segment"].cuda(non_blocking=True).long()
        logit_sum = torch.zeros(
            (full_count, num_classes), device="cuda", dtype=torch.float32
        )
        logit_count = torch.zeros(full_count, device="cuda", dtype=torch.float32)
        single_counts = []
        with torch.no_grad():
            for mask_idx, selected_index in enumerate(masks):
                seed_scene_transform(
                    f"segmentor:{args.split}:{scene_name}:m{mask_idx}", args.seed
                )
                selected_index = np.asarray(selected_index, dtype=np.int64)
                single_counts.append(int(selected_index.shape[0]))
                mask_totals[mask_idx] += int(selected_index.shape[0])
                model_input = to_cuda_input(segmentor_input_cpu, selected_index)
                context = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if bool(args.amp)
                    else torch.amp.autocast("cuda", enabled=False)
                )
                with context:
                    output = segmentor_model(model_input)
                    sparse_logits = output["eval_sparse_logits"].float()
                index_cuda = torch.as_tensor(
                    selected_index, dtype=torch.long, device="cuda"
                )
                logit_sum.index_add_(0, index_cuda, sparse_logits)
                logit_count.index_add_(
                    0,
                    index_cuda,
                    torch.ones(index_cuda.shape[0], device="cuda", dtype=torch.float32),
                )
                current_prefix = int(mask_idx + 1)
                if current_prefix in prefix_confusions:
                    prefix_union_index = torch.nonzero(
                        logit_count > 0, as_tuple=False
                    ).flatten()
                    prefix_union_logits = logit_sum[prefix_union_index] / logit_count[
                        prefix_union_index
                    ].unsqueeze(-1)
                    prefix_union_coord = full_coord[prefix_union_index]
                    prefix_union_segment = full_segment[prefix_union_index]
                    add_union_metrics(
                        prefix_confusions[current_prefix],
                        prefix_union_logits,
                        prefix_union_coord,
                        prefix_union_segment,
                        full_coord,
                        full_segment,
                        num_classes,
                        ignore_index,
                        knn_k_list,
                        full_score_modes,
                    )
                    prefix_draw_total = int(np.sum(single_counts))
                    prefix_union_count = int(prefix_union_index.numel())
                    prefix_valid_union = int(
                        (prefix_union_segment != ignore_index).sum().item()
                    )
                    prefix_valid_raw = int((full_segment != ignore_index).sum().item())
                    prefix_row = dict(
                        masks=current_prefix,
                        scene=scene_name,
                        raw=full_count,
                        target_grid004=target_count,
                        single_mean=float(np.mean(single_counts)),
                        single_min=int(np.min(single_counts)),
                        single_max=int(np.max(single_counts)),
                        draw_total=prefix_draw_total,
                        union=prefix_union_count,
                        duplicates=int(prefix_draw_total - prefix_union_count),
                        single_over_target_mean=float(
                            np.mean(single_counts) / max(target_count, 1)
                        ),
                        union_over_target=float(
                            prefix_union_count / max(target_count, 1)
                        ),
                        union_over_raw=float(prefix_union_count / max(full_count, 1)),
                        duplicate_frac_of_draws=float(
                            (prefix_draw_total - prefix_union_count)
                            / max(prefix_draw_total, 1)
                        ),
                        raw_valid=prefix_valid_raw,
                        union_valid=prefix_valid_union,
                        union_valid_over_raw_valid=float(
                            prefix_valid_union / max(prefix_valid_raw, 1)
                        ),
                        union_invalid_frac=float(
                            (prefix_union_count - prefix_valid_union)
                            / max(prefix_union_count, 1)
                        ),
                    )
                    prefix_scene_rows.append(prefix_row)
                    prefix_totals[current_prefix]["raw"] += full_count
                    prefix_totals[current_prefix]["target"] += int(target_count)
                    prefix_totals[current_prefix]["draw_total"] += prefix_draw_total
                    prefix_totals[current_prefix]["union"] += prefix_union_count
                    prefix_totals[current_prefix]["valid_union"] += prefix_valid_union
                    prefix_totals[current_prefix]["valid_raw"] += prefix_valid_raw
        union_index = torch.nonzero(logit_count > 0, as_tuple=False).flatten()
        union_logits = logit_sum[union_index] / logit_count[union_index].unsqueeze(-1)
        union_coord = full_coord[union_index]
        union_segment = full_segment[union_index]
        add_union_metrics(
            confusions,
            union_logits,
            union_coord,
            union_segment,
            full_coord,
            full_segment,
            num_classes,
            ignore_index,
            knn_k_list,
            full_score_modes,
        )
        draw_total = int(np.sum(single_counts))
        union_count = int(union_index.numel())
        valid_union = int((union_segment != ignore_index).sum().item())
        valid_raw = int((full_segment != ignore_index).sum().item())
        row = dict(
            scene=scene_name,
            raw=full_count,
            target_grid004=target_count,
            single_mean=float(np.mean(single_counts)),
            single_min=int(np.min(single_counts)),
            single_max=int(np.max(single_counts)),
            draw_total=draw_total,
            union=union_count,
            duplicates=int(draw_total - union_count),
            single_over_target_mean=float(
                np.mean(single_counts) / max(target_count, 1)
            ),
            union_over_target=float(union_count / max(target_count, 1)),
            union_over_raw=float(union_count / max(full_count, 1)),
            duplicate_frac_of_draws=float(
                (draw_total - union_count) / max(draw_total, 1)
            ),
            raw_valid=valid_raw,
            union_valid=valid_union,
            union_valid_over_raw_valid=float(valid_union / max(valid_raw, 1)),
            union_invalid_frac=float((union_count - valid_union) / max(union_count, 1)),
        )
        scene_rows.append(row)
        totals["raw"] += full_count
        totals["target"] += int(target_count)
        totals["draw_total"] += draw_total
        totals["union"] += union_count
        totals["valid_union"] += valid_union
        totals["valid_raw"] += valid_raw
        if (
            scene_pos == 0
            or scene_pos + 1 == len(scene_names)
            or (scene_pos + 1) % int(args.print_freq) == 0
        ):
            metrics = {
                name: metric_from_confusion(conf) for name, conf in confusions.items()
            }
            primary_full_name = full_metric_name(knn_k_list[-1], full_score_modes[0])
            elapsed = time.time() - start_time
            print(
                f"done={scene_pos + 1}/{len(scene_names)} scene={scene_name} union/target={row['union_over_target']:.3f} union/raw={row['union_over_raw']:.3f} main={metrics['main']['mIoU']:.4f} subset={metrics['subset_sparse']['mIoU']:.4f} {primary_full_name}={metrics[primary_full_name]['mIoU']:.4f} elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )
    metrics = {name: metric_from_confusion(conf) for name, conf in confusions.items()}
    scene_csv = output_root / "scene_metrics.csv"
    with scene_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scene_rows)
    prefix_scene_csv = output_root / "prefix_scene_metrics.csv"
    with prefix_scene_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prefix_scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prefix_scene_rows)
    for name, conf in confusions.items():
        np.save(output_root / f"confusion_{name}.npy", conf)
        np.savetxt(
            output_root / f"confusion_{name}.csv",
            conf.astype(np.int64),
            fmt="%d",
            delimiter=",",
        )
    prefix_summary_rows = []
    prefix_metrics = {}
    for prefix in prefix_masks:
        prefix = int(prefix)
        current_metrics = {
            name: metric_from_confusion(conf)
            for name, conf in prefix_confusions[prefix].items()
        }
        prefix_metrics[str(prefix)] = current_metrics
        for name, conf in prefix_confusions[prefix].items():
            np.save(output_root / f"prefix{prefix}_confusion_{name}.npy", conf)
            np.savetxt(
                output_root / f"prefix{prefix}_confusion_{name}.csv",
                conf.astype(np.int64),
                fmt="%d",
                delimiter=",",
            )
        current_totals = prefix_totals[prefix]
        prefix_summary_rows.append(
            dict(
                masks=prefix,
                raw_total=current_totals["raw"],
                target_grid004_total=current_totals["target"],
                single_selected_mean_total=float(
                    current_totals["draw_total"] / max(prefix, 1)
                ),
                single_mean_over_target=float(
                    current_totals["draw_total"]
                    / max(prefix * current_totals["target"], 1)
                ),
                draw_total=current_totals["draw_total"],
                union_total=current_totals["union"],
                union_over_target=float(
                    current_totals["union"] / max(current_totals["target"], 1)
                ),
                union_over_raw=float(
                    current_totals["union"] / max(current_totals["raw"], 1)
                ),
                duplicates_total=int(
                    current_totals["draw_total"] - current_totals["union"]
                ),
                duplicate_frac_of_draws=float(
                    (current_totals["draw_total"] - current_totals["union"])
                    / max(current_totals["draw_total"], 1)
                ),
                valid_raw_total=current_totals["valid_raw"],
                union_valid_total=current_totals["valid_union"],
                union_valid_over_raw_valid=float(
                    current_totals["valid_union"] / max(current_totals["valid_raw"], 1)
                ),
                union_invalid_frac=float(
                    (current_totals["union"] - current_totals["valid_union"])
                    / max(current_totals["union"], 1)
                ),
                metrics=current_metrics,
            )
        )
    prefix_summary = dict(
        args=config_payload,
        scenes=len(scene_names),
        prefixes=prefix_masks,
        by_prefix=prefix_summary_rows,
        scene_csv=str(prefix_scene_csv),
    )
    (output_root / "prefix_summary.json").write_text(
        json.dumps(prefix_summary, indent=2)
    )
    prefix_metrics_csv = output_root / "prefix_metrics.csv"
    with prefix_metrics_csv.open("w") as f:
        f.write("masks,metric,mIoU,mAcc,allAcc\n")
        for prefix in prefix_masks:
            for name, result in prefix_metrics[str(int(prefix))].items():
                f.write(
                    f"{int(prefix)},{name},{result['mIoU']:.8f},{result['mAcc']:.8f},{result['allAcc']:.8f}\n"
                )
    prefix_coverage_csv = output_root / "prefix_coverage.csv"
    with prefix_coverage_csv.open("w", newline="") as f:
        fieldnames = [key for key in prefix_summary_rows[0].keys() if key != "metrics"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prefix_summary_rows:
            writer.writerow({key: row[key] for key in fieldnames})
    summary = dict(
        args=config_payload,
        scenes=len(scene_names),
        elapsed_sec=time.time() - start_time,
        raw_total=totals["raw"],
        target_grid004_total=totals["target"],
        single_selected_totals=[int(x) for x in mask_totals.tolist()],
        single_selected_mean_total=float(mask_totals.mean()),
        single_mean_over_target=float(mask_totals.mean() / max(totals["target"], 1)),
        draw_total=totals["draw_total"],
        union_total=totals["union"],
        union_over_target=float(totals["union"] / max(totals["target"], 1)),
        union_over_raw=float(totals["union"] / max(totals["raw"], 1)),
        duplicates_total=int(totals["draw_total"] - totals["union"]),
        duplicate_frac_of_draws=float(
            (totals["draw_total"] - totals["union"]) / max(totals["draw_total"], 1)
        ),
        valid_raw_total=totals["valid_raw"],
        union_valid_total=totals["valid_union"],
        union_valid_over_raw_valid=float(
            totals["valid_union"] / max(totals["valid_raw"], 1)
        ),
        union_invalid_frac=float(
            (totals["union"] - totals["valid_union"]) / max(totals["union"], 1)
        ),
        metrics=metrics,
        scene_csv=str(scene_csv),
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    metrics_csv = output_root / "metrics.csv"
    with metrics_csv.open("w") as f:
        f.write("metric,mIoU,mAcc,allAcc\n")
        for name, result in metrics.items():
            f.write(
                f"{name},{result['mIoU']:.8f},{result['mAcc']:.8f},{result['allAcc']:.8f}\n"
            )
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
