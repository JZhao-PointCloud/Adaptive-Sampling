#!/usr/bin/env python

import argparse
import json
import math
import random
from pathlib import Path


def collect_split_entries(data_root, splits):
    root = Path(data_root)
    split_entries = []
    for split in splits:
        split_path = root / split
        if split_path.is_file():
            with open(split_path, "r") as f:
                entries = json.load(f)
            entries = [Path(entry).as_posix() for entry in entries]
        elif split_path.is_dir():
            entries = [
                path.relative_to(root).as_posix()
                for path in split_path.iterdir()
                if path.is_dir()
            ]
        else:
            raise FileNotFoundError(f"Split path does not exist: {split_path}")
        split_entries.append((split, sorted(entries)))
    return split_entries


def sample_entries(split_entries, ratio, seed, min_count):
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if min_count < 0:
        raise ValueError(f"min_count must be >= 0, got {min_count}")

    rng = random.Random(seed)
    selected = []
    summary = []
    for split, entries in split_entries:
        if not entries:
            summary.append((split, 0, 0))
            continue
        count = min(len(entries), max(min_count, math.ceil(len(entries) * ratio)))
        chosen = sorted(rng.sample(entries, count))
        selected.extend(chosen)
        summary.append((split, len(entries), count))
    return selected, summary


def write_entries(entries, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + ("\n" if entries else ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic relative sample list for dataset lr_file."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    split_entries = collect_split_entries(args.data_root, args.splits)
    selected, summary = sample_entries(
        split_entries=split_entries,
        ratio=args.ratio,
        seed=args.seed,
        min_count=args.min_count,
    )
    write_entries(selected, args.output)

    total = sum(total for _, total, _ in summary)
    picked = sum(count for _, _, count in summary)
    print(
        "Wrote {} / {} entries ({:.2%}) to {}".format(
            picked, total, picked / total if total else 0.0, args.output
        )
    )
    for split, split_total, split_count in summary:
        print(f"{split}: {split_count} / {split_total}")


if __name__ == "__main__":
    main()
