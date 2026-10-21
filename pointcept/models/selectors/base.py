from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass
class SelectorOutput:
    sampled_input: Dict[str, torch.Tensor]
    selected_index: torch.Tensor
    target_counts: torch.Tensor
    total_points: torch.Tensor
    sampler_score: torch.Tensor
    point_prob: torch.Tensor
    surrogate_prob: torch.Tensor
    sampler_weight: Optional[torch.Tensor] = None
    prior_selected: int = 0
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


class BaseSelector(nn.Module):
    """Minimal base for offline, fixed-index selection."""

    def __init__(self, **kwargs):
        super().__init__()

    def step(self):
        pass
