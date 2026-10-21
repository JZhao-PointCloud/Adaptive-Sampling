"""
PartNet Dataset (Unified Part Segmentation)

Expected directory structure:
    data_root/
        train/<category>_<id>/coord.npy
                               segment_level_1.npy
                               segment_level_2.npy (optional)
                               segment_level_3.npy (optional)
        val/<category>_<id>/...
        test/<category>_<id>/...
"""

import glob
import json
import os

import numpy as np
import torch

from pointcept.utils.cache import shared_dict
from pointcept.utils.logger import get_root_logger

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class PartNetDataset(DefaultDataset):
    VALID_ASSETS = ["coord"]

    def __init__(
        self,
        segment_level="max",
        label_meta_file=None,
        label_meta_split="train",
        lr_file=None,
        selected_index_file=None,
        marginal_prob_file=None,
        **kwargs,
    ):
        self.segment_level = self._parse_segment_level(segment_level)
        self.label_meta_split = label_meta_split
        self.lr = (
            np.atleast_1d(np.loadtxt(lr_file, dtype=str)).tolist()
            if lr_file is not None
            else None
        )
        self.selected_index = (
            torch.load(selected_index_file, map_location="cpu", weights_only=False)
            if selected_index_file is not None
            else None
        )
        self.marginal_prob = (
            torch.load(marginal_prob_file, map_location="cpu", weights_only=False)
            if marginal_prob_file is not None
            else None
        )
        data_root = kwargs.get("data_root", "data/dataset")
        if label_meta_file is None:
            label_meta_file = f"partnet_label_meta_{self._segment_level_repr(self.segment_level)}.json"
        self.label_meta_path = (
            label_meta_file
            if os.path.isabs(label_meta_file)
            else os.path.join(data_root, label_meta_file)
        )
        super().__init__(**kwargs)
        label_meta = self._load_or_build_label_meta()
        self.category_offsets = {k: int(v) for k, v in label_meta["offsets"].items()}
        self.category_num_parts = {
            k: int(v) for k, v in label_meta["num_parts"].items()
        }
        self.num_classes = int(label_meta["num_classes"])
        self.class_names = list(label_meta["names"])
        logger = get_root_logger()
        logger.info(
            "PartNet unified labels: {} categories, {} classes, level={}.".format(
                len(self.category_offsets),
                self.num_classes,
                self.segment_level,
            )
        )

    @staticmethod
    def _parse_segment_level(segment_level):
        if isinstance(segment_level, str):
            segment_level = segment_level.strip().lower()
            if segment_level == "max":
                return "max"
            if segment_level.isdigit():
                segment_level = int(segment_level)
            else:
                raise ValueError(
                    "segment_level must be an integer level or 'max', got {}".format(
                        segment_level
                    )
                )
        if isinstance(segment_level, int):
            if segment_level < 1:
                raise ValueError("segment_level must be >= 1 when using integer level.")
            return segment_level
        raise ValueError(
            "segment_level must be an integer level or 'max', got type {}".format(
                type(segment_level)
            )
        )

    @staticmethod
    def _segment_level_repr(segment_level):
        return "max" if segment_level == "max" else f"level_{segment_level}"

    @staticmethod
    def _parse_category(name):
        if "_" not in name:
            raise ValueError(
                "PartNet sample name must follow '<category>_<id>', got {}".format(name)
            )
        return name.rsplit("_", 1)[0]

    def _resolve_segment_asset(self, assets):
        candidates = []
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if not asset.startswith("segment_level_"):
                continue
            level_text = asset[len("segment_level_") : -4]
            if not level_text.isdigit():
                continue
            candidates.append((int(level_text), asset))
        if not candidates:
            raise FileNotFoundError(
                "No segment_level_*.npy is found in current PartNet sample folder."
            )
        if self.segment_level == "max":
            return max(candidates, key=lambda x: x[0])[1]
        for level, asset in candidates:
            if level == self.segment_level:
                return asset
        raise FileNotFoundError(
            "segment_level_{}.npy is not found in current PartNet sample folder.".format(
                self.segment_level
            )
        )

    def _build_label_meta(self):
        split_root = os.path.join(self.data_root, self.label_meta_split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                "PartNet label meta split root not found: {}".format(split_root)
            )
        sample_dirs = sorted(glob.glob(os.path.join(split_root, "*")))
        if len(sample_dirs) == 0:
            raise RuntimeError(
                "No sample folder found in {} for building label meta.".format(
                    split_root
                )
            )
        category_max_label = {}
        for sample_dir in sample_dirs:
            assets = os.listdir(sample_dir)
            segment_asset = self._resolve_segment_asset(assets)
            segment_path = os.path.join(sample_dir, segment_asset)
            segment = (
                np.load(segment_path, mmap_mode="r").reshape([-1]).astype(np.int32)
            )
            valid_mask = segment != self.ignore_index
            if not np.any(valid_mask):
                continue
            sample_name = os.path.basename(sample_dir)
            category = self._parse_category(sample_name)
            max_label = int(segment[valid_mask].max())
            category_max_label[category] = max(
                max_label, category_max_label.get(category, -1)
            )

        categories = sorted(category_max_label.keys())
        if len(categories) == 0:
            raise RuntimeError(
                "No valid labels were found in {} while building label meta.".format(
                    split_root
                )
            )

        num_parts = {
            category: category_max_label[category] + 1 for category in categories
        }
        offsets = {}
        names = []
        offset = 0
        for category in categories:
            offsets[category] = offset
            for local_label in range(num_parts[category]):
                names.append(f"{category}:{local_label}")
            offset += num_parts[category]

        return dict(
            version=1,
            segment_level=self.segment_level,
            label_meta_split=self.label_meta_split,
            categories=categories,
            num_parts=num_parts,
            offsets=offsets,
            num_classes=offset,
            names=names,
        )

    def _load_or_build_label_meta(self):
        logger = get_root_logger()
        level_repr = self._segment_level_repr(self.segment_level)
        if os.path.isfile(self.label_meta_path):
            with open(self.label_meta_path, "r") as f:
                label_meta = json.load(f)
            if (
                label_meta.get("segment_level") == self.segment_level
                and label_meta.get("label_meta_split") == self.label_meta_split
            ):
                return label_meta
            logger.warning(
                "Existing PartNet label meta {} does not match current setting "
                "(level={}, split={}), rebuilding.".format(
                    self.label_meta_path,
                    self.segment_level,
                    self.label_meta_split,
                )
            )

        label_meta = self._build_label_meta()
        os.makedirs(os.path.dirname(self.label_meta_path), exist_ok=True)
        tmp_path = self.label_meta_path + f".tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(label_meta, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.label_meta_path)
            logger.info(
                "PartNet label meta ({}) is written to {}.".format(
                    level_repr, self.label_meta_path
                )
            )
        except OSError as e:
            logger.warning(
                "Failed to persist PartNet label meta to {}: {}".format(
                    self.label_meta_path, e
                )
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return label_meta

    def get_data_list(self):
        if self.lr is None:
            return sorted(super().get_data_list())
        return [os.path.join(self.data_root, name) for name in self.lr]

    def _lookup_aux_data(self, aux_data, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        rel_path = os.path.relpath(data_path, self.data_root).replace(os.sep, "/")
        candidates = (
            self.get_data_name(idx),
            rel_path,
            rel_path.replace("/", "-"),
        )
        for key in candidates:
            if key in aux_data:
                return aux_data[key]
        raise KeyError(
            "Missing auxiliary data for {}. Tried keys: {}".format(
                data_path, candidates
            )
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        assets = os.listdir(data_path)
        if "coord.npy" not in assets:
            raise FileNotFoundError("coord.npy not found in {}".format(data_path))
        segment_asset = self._resolve_segment_asset(assets)

        coord = np.load(os.path.join(data_path, "coord.npy")).astype(np.float32)
        segment = (
            np.load(os.path.join(data_path, segment_asset))
            .reshape([-1])
            .astype(np.int32)
        )
        category = self._parse_category(name)
        if category not in self.category_offsets:
            raise KeyError(
                "Category {} is missing from PartNet label meta {}".format(
                    category, self.label_meta_path
                )
            )
        offset = self.category_offsets[category]
        num_parts = self.category_num_parts[category]
        valid_mask = segment != self.ignore_index
        if np.any(valid_mask):
            local_min = int(segment[valid_mask].min())
            local_max = int(segment[valid_mask].max())
            if local_min < 0 or local_max >= num_parts:
                raise ValueError(
                    "Local label out of range for category {} in {}: "
                    "got min/max={}/{}, valid=[0, {})".format(
                        category, data_path, local_min, local_max, num_parts
                    )
                )
            segment = segment.copy()
            segment[valid_mask] += offset

        data_dict = dict(
            coord=coord,
            segment=segment,
            instance=np.ones(coord.shape[0], dtype=np.int32) * -1,
            name=name,
            split=split,
            category=category,
        )
        if self.selected_index is not None:
            sampled_index = self._lookup_aux_data(self.selected_index, idx)
            if torch.is_tensor(sampled_index):
                sampled_index = sampled_index.cpu().numpy()
            data_dict["sampled_index"] = (
                np.asarray(sampled_index).reshape([-1]).astype(np.int64)
            )
        if self.marginal_prob is not None:
            marginal_prob = self._lookup_aux_data(self.marginal_prob, idx)
            if torch.is_tensor(marginal_prob):
                marginal_prob = marginal_prob.cpu().numpy()
            data_dict["marginal_prob"] = (
                np.asarray(marginal_prob).reshape([-1]).astype(np.float32)
            )
        return data_dict
