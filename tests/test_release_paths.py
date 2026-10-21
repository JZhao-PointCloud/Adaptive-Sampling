"""Regression checks for checkpoint, evaluation population and sampling budgets."""

from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from tools import eval_multimask_sampler_ensemble as ensemble
from tools import precompute_learned_sampler_indices as learned
from tools.precompute_fixed_selector_indices import grid_count


@pytest.mark.parametrize("loader", [learned.load_model, ensemble.load_segmentor])
@pytest.mark.parametrize("valid", [False, True])
def test_checkpoint_loaders_require_matching_weights(
    tmp_path, monkeypatch, loader, valid
):
    class CPUModel(torch.nn.Linear):
        def cuda(self):
            return self

    model = CPUModel(2, 2)
    monkeypatch.setattr("pointcept.models.build_model", lambda cfg: model)
    expected = {
        key: torch.ones_like(value) for key, value in model.state_dict().items()
    }
    state = {f"module.{key}": value for key, value in expected.items()}
    if not valid:
        state = {"unrelated_parameter": torch.tensor(1.0)}
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": state}, checkpoint)
    config = tmp_path / "config.py"
    config.write_text("model = dict(type='CPUModel')\n")
    if not valid:
        with pytest.raises(RuntimeError, match="Missing key"):
            loader(str(config), str(checkpoint))
    else:
        _, loaded = loader(str(config), str(checkpoint))
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert not loaded.training


def test_evaluation_requires_complete_unique_scene_lists():
    dataset = SimpleNamespace(data_list=[0, 1], get_data_name=lambda i: ("b", "a")[i])
    index = ensemble.scene_index(dataset)
    assert ensemble.match_scene_names(index, {"a": 0, "b": 1}) == ["a", "b"]
    with pytest.raises(ValueError, match="missing from sampler"):
        ensemble.match_scene_names({"a": 0}, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="missing from segmentor"):
        ensemble.match_scene_names({"a": 0, "b": 1}, {"a": 0})
    with pytest.raises(ValueError, match="empty"):
        ensemble.match_scene_names({}, {})
    dataset.get_data_name = lambda i: "duplicate"
    with pytest.raises(ValueError, match="Duplicate"):
        ensemble.scene_index(dataset)


def test_scene_seeds_reproduce_rng_after_other_scenes():
    def draw():
        return random.random(), np.random.rand(), torch.randperm(32)

    learned.seed_scene_transform("val:scene_a", 11)
    first = draw()
    learned.seed_scene_transform("val:scene_b", 11)
    draw()
    learned.seed_scene_transform("val:scene_a", 11)
    repeat = draw()
    assert first[:2] == repeat[:2]
    torch.testing.assert_close(first[2], repeat[2])


def test_original_cleaned_count_option_matches_post_cleanup_budget():
    coord = np.repeat(
        np.column_stack([np.arange(32) * 0.1 + 0.001, np.zeros(32), np.zeros(32)]),
        40,
        axis=0,
    ).astype(np.float32)
    budget = grid_count(coord, 0.04)
    args = SimpleNamespace(
        masks=2,
        split="val",
        seed=7,
        calibrate_cleaned_count=True,
        final_grid_cleanup_size=0.02,
    )
    masks = ensemble.generate_masks(args, "toy", coord, np.zeros(len(coord)), budget)
    assert all(len(mask) == budget for mask in masks)
    repeat = ensemble.generate_masks(args, "toy", coord, np.zeros(len(coord)), budget)
    for first, second in zip(masks, repeat):
        np.testing.assert_array_equal(first, second)


def test_prob_mode_preserves_original_softmax_after_logit_averaging(monkeypatch):
    logits = torch.tensor([[10.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]])
    averaged = logits.mean(0, keepdim=True)
    # Only bypass the CUDA KNN kernel; exercise the actual score conversion.
    monkeypatch.setattr(
        ensemble, "interpolate_scores_knn", lambda scores, *args: scores
    )
    actual = ensemble.interpolate_logits_knn(
        averaged, torch.zeros(1, 3), torch.zeros(1, 3), 1, score_mode="prob"
    )
    torch.testing.assert_close(actual, averaged.softmax(-1))
    assert actual.argmax(-1).item() == 0
    assert logits.softmax(-1).mean(0).argmax().item() == 1
