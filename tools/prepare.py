"""Prepare canonical train/val selection caches using the same dataset label mapping."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from pointcept.datasets import build_dataset
from pointcept.utils.config import Config
from tools.precompute_fixed_selector_indices import (
    PrecomputeConfig,
    compute_selected_index,
    compute_marginal_probability,
)


def prepare(
    config, output_root, kind, target_grid_size, base_grid_size, min_points=256
):
    cfg = Config.fromfile(config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    policy = PrecomputeConfig(
        data_root="",
        selector="grid" if kind == "grid" else "oracle_waterfill",
        output_kind="marginal_probability"
        if kind == "probabilities"
        else "selected_index",
        grid_size=base_grid_size,
        target_grid_size=target_grid_size,
        final_grid_cleanup_size=(
            cfg.get("oracle_final_grid_cleanup_size", None)
            if kind == "oracle" else None
        ),
        search_cleanup_budget=(
            cfg.get("oracle_search_cleanup_budget", False)
            if kind == "oracle" else False
        ),
        search_cleanup_max_iter=8,
        search_cleanup_err_tol=1000,
        budget_ratio=1.0,
        min_sample_points=min_points,
        num_classes=cfg.data.num_classes,
        ignore_index=cfg.data.ignore_index,
        include_ignore_as_class=cfg.get("oracle_include_ignore_as_class", False),
    )
    for split in ("train", "val"):
        data_cfg = copy.deepcopy(cfg.data[split])
        data_cfg.pop("selected_index_file", None)
        data_cfg.pop("marginal_prob_file", None)
        data_cfg.update(transform=[], test_mode=False, loop=1)
        dataset = build_dataset(data_cfg)
        if (
            getattr(dataset, "num_classes", cfg.data.num_classes)
            != cfg.data.num_classes
        ):
            raise ValueError("Dataset label space does not match config num_classes")
        records = {}
        for i in range(len(dataset)):
            item = dataset.get_data(i)
            name = dataset.get_data_name(i)
            if name in records:
                raise ValueError(f"Duplicate sample key: {name}")
            function = (
                compute_marginal_probability
                if kind == "probabilities"
                else compute_selected_index
            )
            records[name] = function(
                item["coord"], item["segment"], policy, scene_key=f"{split}:{name}"
            )
            if (i + 1) % 25 == 0:
                print(f"{split}: {i + 1}/{len(dataset)} samples", flush=True)
        suffix = "probabilities" if kind == "probabilities" else "indices"
        torch.save(records, output_root / f"{split}_{suffix}.pth")
        print(f"{split}: {len(records)} samples")
    (output_root / f"{kind}_preparation.json").write_text(
        json.dumps(dict(config=str(config), kind=kind, policy=asdict(policy)), indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--kind", choices=["grid", "oracle", "probabilities"], required=True
    )
    parser.add_argument("--target-grid-size", type=float, required=True)
    parser.add_argument("--base-grid-size", type=float, required=True)
    parser.add_argument("--min-points", type=int, default=256)
    args = parser.parse_args()
    if args.target_grid_size <= 0 or args.base_grid_size <= 0 or args.min_points < 1:
        parser.error("Grid sizes and minimum point count must be positive")
    prepare(
        args.config,
        args.output_root,
        args.kind,
        args.target_grid_size,
        args.base_grid_size,
        args.min_points,
    )


if __name__ == "__main__":
    main()
