"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F


def _mix_local_sampled_index(sampled_index, sampled_index_offset, coord_offset):
    if sampled_index_offset.numel() != coord_offset.numel():
        return sampled_index, sampled_index_offset

    coord_counts = coord_offset.diff(prepend=coord_offset.new_zeros(1))
    index_counts = sampled_index_offset.diff(prepend=sampled_index_offset.new_zeros(1))

    mixed_index = []
    mixed_offset = []
    index_start = 0
    mixed_count = 0
    sample_idx = 0
    num_samples = int(coord_offset.numel())

    while sample_idx < num_samples:
        left_index_count = int(index_counts[sample_idx].item())
        left_point_count = int(coord_counts[sample_idx].item())
        left_index = sampled_index[index_start : index_start + left_index_count]
        index_start += left_index_count
        merged_index = left_index

        if sample_idx + 1 < num_samples:
            right_index_count = int(index_counts[sample_idx + 1].item())
            right_index = (
                sampled_index[index_start : index_start + right_index_count]
                + left_point_count
            )
            index_start += right_index_count
            merged_index = torch.cat([left_index, right_index], dim=0)

        mixed_index.append(merged_index)
        mixed_count += int(merged_index.shape[0])
        mixed_offset.append(mixed_count)
        sample_idx += 2

    if mixed_index:
        return (
            torch.cat(mixed_index, dim=0),
            sampled_index_offset.new_tensor(mixed_offset),
        )
    return sampled_index, sampled_index_offset


def _collate_inverse(batch, key):
    """Shift per-sample inverse maps into the concatenated sparse point space."""
    if key != "inverse" or "coord" not in batch[0]:
        return collate_fn([d[key] for d in batch])

    inverse_list = []
    coord_start = 0
    for data in batch:
        inverse_list.append(data[key] + coord_start)
        coord_start += data["coord"].shape[0]
    return collate_fn(inverse_list)


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        batch = {
            key: (
                (
                    _collate_inverse(batch, key)
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                if "correspondence" not in key
                else collate_fn(
                    [
                        F.pad(
                            d[key].permute(0, 2, 1),
                            (0, max_img_num - d[key].shape[1]),
                            value=-1,
                        ).permute(0, 2, 1)
                        for d in batch
                    ]
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        original_sampled_index = batch.get("sampled_index")
        original_sampled_index_offset = batch.get("sampled_index_offset")
        original_coord_offset = batch.get("offset")
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )
        if (
            original_sampled_index is not None
            and original_sampled_index_offset is not None
            and original_coord_offset is not None
        ):
            (
                batch["sampled_index"],
                batch["sampled_index_offset"],
            ) = _mix_local_sampled_index(
                original_sampled_index,
                original_sampled_index_offset,
                original_coord_offset,
            )
        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
        correspondence_assets = [
            asset for asset in batch.keys() if "correspondence" in asset
        ]
        for correspondence_asset in correspondence_assets:
            offset = batch["offset"]
            start = 0
            N, v, n = batch[correspondence_asset].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones((N, v2, n))
            for i in range(len(offset)):
                if i % 2 == 0:
                    batch_correspondence_mix[start : offset[i], 0:v] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                if i % 2 != 0:
                    batch_correspondence_mix[start : offset[i], v:] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                start = offset[i]
            if len(offset) % 2 == 0:
                pass
            else:
                start = 0 if len(offset) == 1 else offset[-2]
                batch_correspondence_mix[start:N, -v:] = batch[correspondence_asset][
                    start:N, -v:
                ]
            batch[correspondence_asset] = batch_correspondence_mix
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
