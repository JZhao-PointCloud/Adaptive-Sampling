import torch
from pointcept.models.utils import batch2offset, offset2batch


def index_points(input_dict, index):
    num_points = input_dict["coord"].shape[0]
    sampled_input = {}
    for key, value in input_dict.items():
        if key == "offset":
            continue
        if torch.is_tensor(value) and value.ndim > 0 and (value.shape[0] == num_points):
            sampled_input[key] = value[index]
        else:
            sampled_input[key] = value
    batch = offset2batch(input_dict["offset"])[index]
    sampled_input["offset"] = batch2offset(batch)
    return sampled_input
