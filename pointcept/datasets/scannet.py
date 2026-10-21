"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = ["coord", "color", "normal", "segment20", "instance"]
    class2id = np.array(
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    )

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        selected_index_file=None,
        marginal_prob_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = (
            torch.load(la_file, map_location="cpu", weights_only=False)
            if la_file is not None
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
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)
        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        elif self.selected_index is not None:
            sampled_index = self.selected_index[self.get_data_name(idx)]
            if torch.is_tensor(sampled_index):
                sampled_index = sampled_index.cpu().numpy()
            data_dict["sampled_index"] = (
                np.asarray(sampled_index).reshape([-1]).astype(np.int64)
            )
        if self.marginal_prob is not None:
            marginal_prob = self.marginal_prob[self.get_data_name(idx)]
            if torch.is_tensor(marginal_prob):
                marginal_prob = marginal_prob.cpu().numpy()
            data_dict["marginal_prob"] = (
                np.asarray(marginal_prob).reshape([-1]).astype(np.float32)
            )
        return data_dict
