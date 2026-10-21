import importlib.util
from pathlib import Path
import numpy as np

_TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "precompute_fixed_selector_indices.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "precompute_fixed_selector_indices", _TOOL_PATH
)
precompute = importlib.util.module_from_spec(_SPEC)
import sys

sys.modules[_SPEC.name] = precompute
_SPEC.loader.exec_module(precompute)


def test_waterfill_marginal_probability_is_voxel_budget_over_voxel_count():
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    segment = np.zeros(4, dtype=np.int64)
    prob = precompute.waterfill_marginal_probability(
        coord,
        segment,
        target_count=2,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
    )
    np.testing.assert_allclose(prob, np.full(4, 0.5, dtype=np.float32))


def test_waterfill_marginal_probability_can_include_ignore_as_class():
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    segment = np.array([-1, -1, 0, 0], dtype=np.int64)
    default_prob = precompute.waterfill_marginal_probability(
        coord,
        segment,
        target_count=2,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
    )
    ignore_class_prob = precompute.waterfill_marginal_probability(
        coord,
        segment,
        target_count=2,
        num_classes=1,
        ignore_index=-1,
        include_ignore_as_class=True,
        base_grid_size=0.02,
    )
    np.testing.assert_allclose(default_prob, np.array([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_allclose(ignore_class_prob, np.full(4, 0.5, dtype=np.float32))


def test_waterfill_sample_can_include_ignore_as_class():
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    segment = np.array([-1, -1, 0, 0], dtype=np.int64)
    selected = precompute.waterfill_sample(
        coord,
        segment,
        target_count=2,
        num_classes=1,
        ignore_index=-1,
        include_ignore_as_class=True,
        base_grid_size=0.02,
        scene_key="toy:ignore-class",
    )
    assert selected.shape[0] == 2
    assert int((segment[selected] == -1).sum()) == 1
    assert int((segment[selected] == 0).sum()) == 1


def test_waterfill_marginal_bernoulli_uses_marginal_probability():
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    segment = np.zeros(4, dtype=np.int64)
    scene_key = "toy:bernoulli"
    target_count = 2
    selected = precompute.waterfill_marginal_bernoulli(
        coord,
        segment,
        target_count=target_count,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
        scene_key=scene_key,
    )
    prob = precompute.waterfill_marginal_probability(
        coord,
        segment,
        target_count=target_count,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
    )
    rng = precompute.deterministic_rng(scene_key, target_count)
    expected = np.nonzero(rng.random(coord.shape[0]) < prob)[0].astype(np.int64)
    np.testing.assert_array_equal(selected, np.sort(expected))


def test_waterfill_marginal_bernoulli_is_reproducible():
    coord = np.stack(
        [np.arange(20, dtype=np.float32), np.zeros(20), np.zeros(20)], axis=1
    )
    segment = np.zeros(20, dtype=np.int64)
    first = precompute.waterfill_marginal_bernoulli(
        coord,
        segment,
        target_count=10,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
        scene_key="toy:repeat",
    )
    second = precompute.waterfill_marginal_bernoulli(
        coord,
        segment,
        target_count=10,
        num_classes=1,
        ignore_index=-1,
        base_grid_size=0.02,
        scene_key="toy:repeat",
    )
    np.testing.assert_array_equal(first, second)


def test_compute_marginal_probability_uses_resolved_grid_budget():
    coord = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        dtype=np.float32,
    )
    segment = np.zeros(4, dtype=np.int64)
    cfg = precompute.PrecomputeConfig(
        data_root="unused",
        selector="oracle_waterfill",
        output_kind="marginal_probability",
        grid_size=0.02,
        target_grid_size=0.2,
        final_grid_cleanup_size=None,
        search_cleanup_budget=False,
        search_cleanup_max_iter=8,
        search_cleanup_err_tol=1000,
        budget_ratio=1.0,
        min_sample_points=2,
        num_classes=1,
        ignore_index=-1,
    )
    prob = precompute.compute_marginal_probability(
        coord, segment, cfg, scene_key="toy:prob"
    )
    np.testing.assert_allclose(prob, np.full(4, 0.5, dtype=np.float32))
