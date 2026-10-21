import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute fixed selector indices or marginal probabilities"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--output-name", default=None, help="Output stem, e.g. indices or probabilities"
    )
    parser.add_argument(
        "--selector",
        required=True,
        choices=["grid", "oracle_waterfill", "oracle_waterfill_marginal_bernoulli"],
    )
    parser.add_argument(
        "--output-kind",
        default="selected_index",
        choices=["selected_index", "marginal_probability"],
        help="Save selected indices or per-point oracle waterfill marginal probabilities.",
    )
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument(
        "--target-grid-size",
        type=float,
        default=None,
        help="Optional final target budget reference grid size, e.g. 0.04 for old aim004.",
    )
    parser.add_argument(
        "--final-grid-cleanup-size",
        type=float,
        default=None,
        help="Optional post-selection grid cleanup size, e.g. 0.02 for old grid002 cleanup.",
    )
    parser.add_argument(
        "--search-cleanup-budget",
        action="store_true",
        help="Search a pre-cleanup budget so that the final cleaned subset best matches the target budget.",
    )
    parser.add_argument(
        "--search-cleanup-max-iter",
        type=int,
        default=8,
        help="Maximum binary-search iterations when search_cleanup_budget is enabled.",
    )
    parser.add_argument(
        "--search-cleanup-err-tol",
        type=int,
        default=1000,
        help="If the cleaned-count error is within this tolerance, stop local search early.",
    )
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--min-sample-points", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-index", type=int, default=-1)
    parser.add_argument(
        "--include-ignore-as-class",
        action="store_true",
        help="Treat ignore_index points as an extra waterfill class instead of forcing their marginal probability to zero.",
    )
    parser.add_argument(
        "--segment-asset",
        default="auto",
        help="Segmentation asset to load. Use auto, segment20, segment, segment_level_max, or an explicit .npy stem.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--scene-list",
        default=None,
        help="Optional train subset file like limit0.1.txt",
    )
    return parser.parse_args()


@dataclass
class PrecomputeConfig:
    data_root: str
    selector: str
    output_kind: str
    grid_size: float
    target_grid_size: float | None
    final_grid_cleanup_size: float | None
    search_cleanup_budget: bool
    search_cleanup_max_iter: int
    search_cleanup_err_tol: int
    budget_ratio: float
    min_sample_points: int
    num_classes: int
    ignore_index: int
    segment_asset: str = "auto"
    include_ignore_as_class: bool = False


def resolve_segment_path(scene_dir, segment_asset="auto"):
    scene_dir = Path(scene_dir)
    segment_asset = str(segment_asset)
    if segment_asset == "auto":
        for filename in ("segment20.npy", "segment.npy", "segment200.npy"):
            candidate = scene_dir / filename
            if candidate.is_file():
                return candidate
        segment_level_paths = sorted(scene_dir.glob("segment_level_*.npy"))
        if segment_level_paths:
            return max(
                segment_level_paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1])
            )
    elif segment_asset == "segment_level_max":
        segment_level_paths = sorted(scene_dir.glob("segment_level_*.npy"))
        if segment_level_paths:
            return max(
                segment_level_paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1])
            )
    else:
        filename = (
            segment_asset if segment_asset.endswith(".npy") else f"{segment_asset}.npy"
        )
        candidate = scene_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No segment asset '{}' found in {}".format(segment_asset, scene_dir)
    )


def load_scene(scene_dir, segment_asset="auto"):
    coord = np.load(scene_dir / "coord.npy").astype(np.float32, copy=False)
    segment_path = resolve_segment_path(scene_dir, segment_asset)
    segment = np.load(segment_path).reshape(-1).astype(np.int64, copy=False)
    return (coord, segment)


def grid_idx(coord: np.ndarray, grid_size: float) -> np.ndarray:
    if coord.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    grid_coord = np.floor(coord / float(grid_size)).astype(np.int64)
    _, index = np.unique(grid_coord, axis=0, return_index=True)
    return np.sort(index.astype(np.int64, copy=False))


def grid_count(coord: np.ndarray, grid_size: float) -> int:
    if coord.shape[0] == 0:
        return 0
    grid_coord = np.floor(coord / float(grid_size)).astype(np.int64)
    return int(np.unique(grid_coord, axis=0).shape[0])


def capped_waterfill(caps: np.ndarray, target_sum: int) -> np.ndarray:
    caps = np.asarray(caps, dtype=np.int64)
    total = int(caps.sum())
    if target_sum <= 0:
        return np.zeros_like(caps)
    if target_sum >= total:
        return caps.copy()
    lo, hi = (0.0, float(caps.max()))
    for _ in range(50):
        tau = (lo + hi) / 2.0
        current_sum = np.minimum(caps, tau).sum()
        if current_sum >= target_sum:
            hi = tau
        else:
            lo = tau
    clipped = np.minimum(caps, hi)
    result = np.floor(clipped).astype(np.int64)
    remain = int(target_sum - result.sum())
    if remain > 0:
        fractional = clipped - result
        for idx in np.argsort(-fractional):
            if remain == 0:
                break
            if result[idx] < caps[idx]:
                result[idx] += 1
                remain -= 1
    if remain > 0:
        for idx in range(result.shape[0]):
            if remain == 0:
                break
            if result[idx] < caps[idx]:
                result[idx] += 1
                remain -= 1
    return result


def stable_scene_seed(scene_key: str) -> int:
    value = 2166136261
    for ch in scene_key.encode("utf-8"):
        value ^= ch
        value = value * 16777619 & 4294967295
    return int(value)


def deterministic_rng(scene_key: str, budget: int) -> np.random.Generator:
    seed = (
        stable_scene_seed(scene_key) * 1664525 + int(budget) * 1013904223 & 4294967295
    )
    return np.random.default_rng(int(seed))


def waterfill_candidate_index(
    segment: np.ndarray, *, ignore_index: int, include_ignore_as_class: bool
) -> np.ndarray:
    if include_ignore_as_class:
        return np.arange(segment.shape[0], dtype=np.int64)
    if int(ignore_index) < 0:
        return np.where(segment >= 0)[0]
    return np.where(segment != int(ignore_index))[0]


def waterfill_sample(
    coord: np.ndarray,
    segment: np.ndarray,
    target_count: int,
    *,
    num_classes: int,
    ignore_index: int,
    include_ignore_as_class: bool = False,
    base_grid_size: float,
    scene_key: str,
) -> np.ndarray:
    num_points = int(coord.shape[0])
    if target_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if target_count >= num_points:
        return np.arange(num_points, dtype=np.int64)
    rng = deterministic_rng(scene_key, target_count)
    valid_idx = waterfill_candidate_index(
        segment,
        ignore_index=ignore_index,
        include_ignore_as_class=include_ignore_as_class,
    )
    if valid_idx.size == 0:
        return np.sort(
            rng.choice(np.arange(num_points), size=target_count, replace=False)
        )
    valid_target = min(int(target_count), int(valid_idx.size))
    coord_valid = coord[valid_idx]
    segment_valid = segment[valid_idx]
    classes, inv = np.unique(segment_valid, return_inverse=True)
    class_bins_local = [np.where(inv == ci)[0] for ci in range(classes.shape[0])]
    class_counts = np.array(
        [local_idx.shape[0] for local_idx in class_bins_local], dtype=np.int64
    )
    class_budget = capped_waterfill(class_counts, valid_target)
    selected_parts = []
    for class_local_idx, current_class_budget in zip(class_bins_local, class_budget):
        current_class_budget = int(current_class_budget)
        if current_class_budget <= 0:
            continue
        current_global_idx = valid_idx[class_local_idx]
        current_coord = coord[current_global_idx]
        voxel_size = class_spatial_voxel_size(
            current_coord, current_class_budget, base_grid_size=base_grid_size
        )
        coord_min = current_coord.min(axis=0)
        voxel = np.floor((current_coord - coord_min) / voxel_size).astype(np.int64)
        _, voxel_inv, voxel_counts = np.unique(
            voxel, axis=0, return_inverse=True, return_counts=True
        )
        voxel_budget = capped_waterfill(
            voxel_counts.astype(np.int64), current_class_budget
        )
        for voxel_id, current_voxel_budget in enumerate(voxel_budget):
            current_voxel_budget = int(current_voxel_budget)
            if current_voxel_budget <= 0:
                continue
            voxel_local = np.where(voxel_inv == voxel_id)[0]
            if current_voxel_budget >= voxel_local.shape[0]:
                chosen_local = voxel_local
            else:
                chosen_local = rng.choice(
                    voxel_local, size=current_voxel_budget, replace=False
                )
            selected_parts.append(current_global_idx[chosen_local])
    selected = (
        np.unique(np.concatenate(selected_parts, axis=0)).astype(np.int64, copy=False)
        if selected_parts
        else np.empty((0,), dtype=np.int64)
    )
    if selected.shape[0] > target_count:
        selected = rng.choice(selected, size=target_count, replace=False)
    elif selected.shape[0] < target_count:
        mask = np.ones(num_points, dtype=bool)
        mask[selected] = False
        pool = np.where(mask)[0]
        backfill = rng.choice(
            pool, size=target_count - selected.shape[0], replace=False
        )
        selected = np.concatenate([selected, backfill], axis=0)
    return np.sort(selected.astype(np.int64, copy=False))


def waterfill_marginal_probability(
    coord: np.ndarray,
    segment: np.ndarray,
    target_count: int,
    *,
    num_classes: int,
    ignore_index: int,
    include_ignore_as_class: bool = False,
    base_grid_size: float,
) -> np.ndarray:
    """Return per-point marginal keep probability implied by waterfill sampling.

    Waterfill's only random step is uniform k-of-n sampling inside each final voxel.
    Once class and voxel budgets are fixed, each point in a voxel has marginal
    probability k/n.
    """
    num_points = int(coord.shape[0])
    probability = np.zeros(num_points, dtype=np.float32)
    if target_count <= 0 or num_points == 0:
        return probability
    if target_count >= num_points:
        probability.fill(1.0)
        return probability
    valid_idx = waterfill_candidate_index(
        segment,
        ignore_index=ignore_index,
        include_ignore_as_class=include_ignore_as_class,
    )
    if valid_idx.size == 0:
        return probability
    valid_target = min(int(target_count), int(valid_idx.size))
    segment_valid = segment[valid_idx]
    classes, inv = np.unique(segment_valid, return_inverse=True)
    class_bins_local = [np.where(inv == ci)[0] for ci in range(classes.shape[0])]
    class_counts = np.array(
        [local_idx.shape[0] for local_idx in class_bins_local], dtype=np.int64
    )
    class_budget = capped_waterfill(class_counts, valid_target)
    for class_local_idx, current_class_budget in zip(class_bins_local, class_budget):
        current_class_budget = int(current_class_budget)
        if current_class_budget <= 0:
            continue
        current_global_idx = valid_idx[class_local_idx]
        current_coord = coord[current_global_idx]
        voxel_size = class_spatial_voxel_size(
            current_coord, current_class_budget, base_grid_size=base_grid_size
        )
        coord_min = current_coord.min(axis=0)
        voxel = np.floor((current_coord - coord_min) / voxel_size).astype(np.int64)
        _, voxel_inv, voxel_counts = np.unique(
            voxel, axis=0, return_inverse=True, return_counts=True
        )
        voxel_budget = capped_waterfill(
            voxel_counts.astype(np.int64), current_class_budget
        )
        for voxel_id, current_voxel_budget in enumerate(voxel_budget):
            voxel_local = np.where(voxel_inv == voxel_id)[0]
            voxel_count = int(voxel_local.shape[0])
            if voxel_count <= 0:
                continue
            p_keep = min(1.0, max(0.0, float(current_voxel_budget) / voxel_count))
            probability[current_global_idx[voxel_local]] = p_keep
    return probability


def deterministic_tie_key(num_points: int, scene_key: str, budget: int) -> np.ndarray:
    rng = deterministic_rng(scene_key, budget)
    return rng.random(int(num_points), dtype=np.float64)


def waterfill_marginal_bernoulli(
    coord: np.ndarray,
    segment: np.ndarray,
    target_count: int,
    *,
    num_classes: int,
    ignore_index: int,
    include_ignore_as_class: bool = False,
    base_grid_size: float,
    scene_key: str,
) -> np.ndarray:
    num_points = int(coord.shape[0])
    if target_count <= 0 or num_points == 0:
        return np.empty((0,), dtype=np.int64)
    if target_count >= num_points:
        return np.arange(num_points, dtype=np.int64)
    probability = waterfill_marginal_probability(
        coord,
        segment,
        target_count,
        num_classes=num_classes,
        ignore_index=ignore_index,
        include_ignore_as_class=include_ignore_as_class,
        base_grid_size=base_grid_size,
    )
    rng = deterministic_rng(scene_key, target_count)
    selected = np.nonzero(rng.random(num_points) < probability)[0]
    return np.sort(selected.astype(np.int64, copy=False))


def apply_final_cleanup(
    coord: np.ndarray, selected_index: np.ndarray, cleanup_grid_size: float | None
) -> np.ndarray:
    if cleanup_grid_size is None:
        return selected_index.astype(np.int64, copy=False)
    if selected_index.size == 0:
        return selected_index.astype(np.int64, copy=False)
    cleanup_local = grid_idx(coord[selected_index], cleanup_grid_size)
    return selected_index[cleanup_local].astype(np.int64, copy=False)


def resolve_target_count(coord: np.ndarray, cfg: PrecomputeConfig) -> int:
    target_grid_size = (
        float(cfg.target_grid_size)
        if cfg.target_grid_size is not None
        else float(cfg.grid_size)
    )
    target_count = grid_count(coord, target_grid_size)
    target_count = int(np.ceil(target_count * float(cfg.budget_ratio)))
    target_count = max(int(cfg.min_sample_points), target_count)
    target_count = min(int(coord.shape[0]), target_count)
    return int(target_count)


def select_raw_index(
    coord: np.ndarray,
    segment: np.ndarray,
    raw_budget: int,
    cfg: PrecomputeConfig,
    *,
    scene_key: str,
) -> np.ndarray:
    if cfg.selector == "grid":
        return grid_idx(coord, cfg.target_grid_size or cfg.grid_size)
    if cfg.selector == "oracle_waterfill_marginal_bernoulli":
        return waterfill_marginal_bernoulli(
            coord,
            segment,
            raw_budget,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
            include_ignore_as_class=cfg.include_ignore_as_class,
            base_grid_size=cfg.grid_size,
            scene_key=scene_key,
        )
    return waterfill_sample(
        coord,
        segment,
        raw_budget,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
        include_ignore_as_class=cfg.include_ignore_as_class,
        base_grid_size=cfg.grid_size,
        scene_key=scene_key,
    )


def compute_selected_index(
    coord: np.ndarray, segment: np.ndarray, cfg: PrecomputeConfig, *, scene_key: str
) -> np.ndarray:
    target_count = resolve_target_count(coord, cfg)
    if cfg.search_cleanup_budget and cfg.final_grid_cleanup_size is not None:
        num_points = int(coord.shape[0])
        lo = max(int(target_count), 1)
        hi = num_points
        cache = {}
        best_budget = lo
        best_selected = None
        best_diff = 10**18

        def evaluate_budget(raw_budget: int):
            raw_budget = int(raw_budget)
            if raw_budget not in cache:
                raw_selected = select_raw_index(
                    coord, segment, raw_budget, cfg, scene_key=scene_key
                )
                cleaned = apply_final_cleanup(
                    coord, raw_selected, cfg.final_grid_cleanup_size
                )
                cache[raw_budget] = cleaned
            return cache[raw_budget]

        seen = set()
        for _ in range(max(1, int(cfg.search_cleanup_max_iter))):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            if mid in seen:
                break
            seen.add(mid)
            cleaned = evaluate_budget(mid)
            cleaned_count = int(cleaned.shape[0])
            diff = abs(cleaned_count - int(target_count))
            if diff < best_diff:
                best_diff = diff
                best_budget = int(mid)
                best_selected = cleaned
            if int(best_diff) <= int(cfg.search_cleanup_err_tol):
                break
            if cleaned_count < int(target_count):
                lo = mid + 1
            else:
                hi = mid - 1
        if int(best_diff) > int(cfg.search_cleanup_err_tol):
            candidate_budgets = {
                best_budget,
                max(1, best_budget - 1000),
                min(num_points, best_budget + 1000),
                max(1, lo),
                min(num_points, hi),
            }
            for delta in (2000, 4000, 8000, 16000):
                candidate_budgets.add(max(1, best_budget - delta))
                candidate_budgets.add(min(num_points, best_budget + delta))
            for candidate in sorted(candidate_budgets):
                if int(candidate) in seen:
                    continue
                seen.add(int(candidate))
                cleaned = evaluate_budget(int(candidate))
                cleaned_count = int(cleaned.shape[0])
                diff = abs(cleaned_count - int(target_count))
                if diff < best_diff:
                    best_diff = diff
                    best_budget = int(candidate)
                    best_selected = cleaned
        return np.asarray(best_selected, dtype=np.int64)
    selected_index = select_raw_index(
        coord, segment, target_count, cfg, scene_key=scene_key
    )
    return apply_final_cleanup(coord, selected_index, cfg.final_grid_cleanup_size)


def compute_marginal_probability(
    coord: np.ndarray, segment: np.ndarray, cfg: PrecomputeConfig, *, scene_key: str
) -> np.ndarray:
    if not str(cfg.selector).startswith("oracle_waterfill"):
        raise ValueError(
            "marginal_probability output is only defined for oracle_waterfill selectors"
        )
    target_count = resolve_target_count(coord, cfg)
    return waterfill_marginal_probability(
        coord,
        segment,
        target_count,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
        include_ignore_as_class=cfg.include_ignore_as_class,
        base_grid_size=cfg.grid_size,
    )


def compute_scene_output(split_name, scene_name, scene_dir, cfg):
    coord, segment = load_scene(scene_dir, cfg.segment_asset)
    scene_key = f"{split_name}:{scene_name}"
    if cfg.output_kind == "marginal_probability":
        return (
            scene_name,
            compute_marginal_probability(coord, segment, cfg, scene_key=scene_key),
        )
    return (
        scene_name,
        compute_selected_index(coord, segment, cfg, scene_key=scene_key),
    )


def output_suffix(cfg):
    parts = [cfg.selector]
    if cfg.output_kind == "marginal_probability":
        parts.append("prob")
    if cfg.target_grid_size is None:
        parts.append(f"gridref_g{str(cfg.grid_size).replace('.', '')}")
    else:
        parts.append(f"aimg{str(cfg.target_grid_size).replace('.', '')}")
    if cfg.final_grid_cleanup_size is not None:
        parts.append(f"cleang{str(cfg.final_grid_cleanup_size).replace('.', '')}")
    if cfg.search_cleanup_budget:
        parts.append("search")
    if cfg.include_ignore_as_class:
        parts.append("ignorecls")
    return "_".join(parts)


def build_precompute_config(args):
    return PrecomputeConfig(
        data_root=str(args.data_root),
        selector=str(args.selector),
        output_kind=str(args.output_kind),
        grid_size=float(args.grid_size),
        target_grid_size=None
        if args.target_grid_size is None
        else float(args.target_grid_size),
        final_grid_cleanup_size=None
        if args.final_grid_cleanup_size is None
        else float(args.final_grid_cleanup_size),
        search_cleanup_budget=bool(args.search_cleanup_budget),
        search_cleanup_max_iter=int(args.search_cleanup_max_iter),
        search_cleanup_err_tol=int(args.search_cleanup_err_tol),
        budget_ratio=float(args.budget_ratio),
        min_sample_points=int(args.min_sample_points),
        num_classes=int(args.num_classes),
        ignore_index=int(args.ignore_index),
        include_ignore_as_class=bool(args.include_ignore_as_class),
        segment_asset=str(args.segment_asset),
    )


def compute_scene_selected_index(split_name, scene_name, scene_dir, cfg):
    return compute_scene_output(split_name, scene_name, scene_dir, cfg)


def compute_scene_selected_index_worker(task):
    split_name, scene_name, scene_dir_str, cfg = task
    return compute_scene_selected_index(
        split_name, scene_name, Path(scene_dir_str), cfg
    )


def format_scene_key(split, rel_path):
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0].startswith("Area_"):
        return f"{parts[0]}-{parts[-1]}"
    return parts[-1]


def resolve_scene_dirs(data_root, split, scene_list_path=None):
    data_root = Path(data_root)
    if split == "train" and scene_list_path is not None:
        scene_entries = [
            line.strip()
            for line in Path(scene_list_path).read_text().splitlines()
            if line.strip()
        ]
        result = []
        for entry in scene_entries:
            rel_path = Path(entry)
            if len(rel_path.parts) == 1:
                scene_path = data_root / split / rel_path
                key_path = rel_path
            else:
                scene_path = data_root / rel_path
                key_path = rel_path
            result.append((format_scene_key(split, key_path), scene_path))
        return result
    split_root = data_root / split
    return sorted(
        (
            (format_scene_key(split, Path(split) / path.name), path)
            for path in split_root.iterdir()
            if path.is_dir()
        )
    )


def main():
    args = parse_args()
    cfg = build_precompute_config(args)
    if cfg.output_kind == "marginal_probability" and (
        cfg.final_grid_cleanup_size is not None or cfg.search_cleanup_budget
    ):
        print(
            "[warn] marginal_probability is computed before final cleanup/search; cleanup flags only affect selected_index outputs.",
            flush=True,
        )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        scene_dirs = resolve_scene_dirs(
            args.data_root, split, args.scene_list if split == "train" else None
        )
        selected_index_dict = {}
        start_time = time.time()
        num_workers = max(1, int(args.num_workers))
        if num_workers == 1:
            for scene_name, scene_dir in scene_dirs:
                _, selected_index = compute_scene_selected_index(
                    split, scene_name, scene_dir, cfg
                )
                selected_index_dict[scene_name] = selected_index
        else:
            max_workers = min(
                num_workers, len(scene_dirs), os.cpu_count() or num_workers
            )
            print(
                f"[{split}] precompute workers {max_workers}/{len(scene_dirs)} scenes",
                flush=True,
            )
            tasks = [
                (split, scene_name, str(scene_dir), cfg)
                for scene_name, scene_dir in scene_dirs
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(compute_scene_selected_index_worker, task): task[1]
                    for task in tasks
                }
                completed = 0
                total = len(tasks)
                for future in as_completed(futures):
                    scene_name, selected_index = future.result()
                    selected_index_dict[scene_name] = selected_index
                    completed += 1
                    if completed == 1 or completed == total or completed % 10 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"[{split}] done={completed}/{total} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
        output_path = (
            output_root / f"{split}_{args.output_name or output_suffix(cfg)}.pth"
        )
        torch.save(selected_index_dict, output_path)
        print(output_path)


def class_spatial_voxel_size(
    coord: np.ndarray, target_count: int, base_grid_size: float
) -> float:
    if target_count <= 0 or coord.shape[0] == 0:
        return float(base_grid_size)
    coord_min = coord.min(axis=0)
    coord_max = coord.max(axis=0)
    side = np.maximum(coord_max - coord_min, 1e-12)
    volume = float(side[0] * side[1] * side[2])
    delta = (
        float((volume / float(target_count)) ** (1.0 / 3.0))
        if target_count > 0
        else 1e-12
    )
    if not np.isfinite(delta) or delta <= 0:
        delta = 1e-12
    return float(max(float(base_grid_size), delta))


if __name__ == "__main__":
    main()
