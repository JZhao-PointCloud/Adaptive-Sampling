"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
    def __init__(
        self,
        lr_file=None,
        la_file=None,
        selected_index_file=None,
        marginal_prob_file=None,
        **kwargs,
    ):
        self.lr = (
            np.atleast_1d(np.loadtxt(lr_file, dtype=str)).tolist()
            if lr_file is not None
            else None
        )
        self.la = torch.load(la_file) if la_file is not None else None
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
            data_list = [os.path.join(self.data_root, name) for name in self.lr]
        return data_list

    def _lookup_aux_data(self, aux_data, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        rel_path = os.path.relpath(data_path, self.data_root).replace(os.sep, "/")
        candidates = (
            self.get_data_name(idx),
            rel_path,
            os.path.basename(data_path),
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
        data_dict = super().get_data(idx)
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

    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"
