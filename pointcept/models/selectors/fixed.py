import torch
from .base import BaseSelector, SelectorOutput
from .builder import SELECTORS
from .common import index_points


@SELECTORS.register_module()
class FixedSelector(BaseSelector):
    def __init__(
        self,
        sampled_index_key="sampled_index",
        sampled_index_offset_key="sampled_index_offset",
        **kwargs,
    ):
        super().__init__(
            sample_ratio=1.0,
            min_sample_points=1,
            max_sample_points=None,
            sample_budget_mode="ratio",
            budget_grid_size=None,
            budget_ratio=1.0,
            sampler_algorithm="fixed",
            sampler_eps=1e-06,
            surrogate_prob_mode="point",
        )
        self.sampled_index_key = sampled_index_key
        self.sampled_index_offset_key = sampled_index_offset_key
        self.num_classes = int(kwargs.get("num_classes", 0))
        self.ignore_index = int(kwargs.get("ignore_index", -1))

    def _resolve_selected_index(self, input_dict):
        if self.sampled_index_key not in input_dict:
            raise KeyError(f"{self.sampled_index_key} is required for FixedSelector")
        sampled_index = input_dict[self.sampled_index_key].long()
        if self.sampled_index_offset_key not in input_dict:
            target_counts = sampled_index.new_tensor([sampled_index.shape[0]])
            return (sampled_index, target_counts)
        index_offset = input_dict[self.sampled_index_offset_key].long()
        coord_offset = input_dict["offset"].long()
        selected = []
        target_counts = []
        start = 0
        coord_start = 0
        for index_end, coord_end in zip(index_offset.tolist(), coord_offset.tolist()):
            current_index = sampled_index[start:index_end] + coord_start
            selected.append(current_index)
            target_counts.append(index_end - start)
            start = index_end
            coord_start = coord_end
        return (torch.cat(selected, dim=0), sampled_index.new_tensor(target_counts))

    def forward(self, input_dict):
        self.step()
        device = input_dict["coord"].device
        selected_index, target_counts = self._resolve_selected_index(input_dict)
        total_points = input_dict["offset"].diff(
            prepend=input_dict["offset"].new_zeros(1)
        )
        num_points = int(input_dict["coord"].shape[0])
        dummy = torch.zeros(num_points, device=device, dtype=torch.float32)
        selected_count = float(selected_index.numel())
        total_count = float(num_points)
        aux_metrics = dict(
            sample_count=dummy.new_tensor(selected_count),
            target_sample_count=dummy.new_tensor(selected_count),
            sample_ratio_actual=dummy.new_tensor(
                selected_count / max(total_count, 1.0)
            ),
            target_sample_ratio=dummy.new_tensor(
                selected_count / max(total_count, 1.0)
            ),
        )
        sampled_input = index_points(input_dict, selected_index)
        return SelectorOutput(
            sampled_input=sampled_input,
            selected_index=selected_index,
            target_counts=target_counts,
            total_points=total_points,
            sampler_score=dummy,
            point_prob=dummy,
            surrogate_prob=dummy,
            aux_metrics=aux_metrics,
        )
