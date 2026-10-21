from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pytest
import torch

from pointcept.utils.config import Config
from tools.precompute_learned_sampler_indices import (
    calibrate_probability_to_target_count,
    sample_predicted_probability,
)
from tools.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_all_public_configs_resolve_without_historical_files():
    for path in (ROOT / "configs").rglob("*.py"):
        cfg = Config.fromfile(str(path))
        assert cfg.model.type in (
            "SelectorSegmentor-v1m1", "MarginalSamplerRegressor", "DefaultSegmentorV2"
        )
        assert cfg.epoch % cfg.eval_epoch == 0
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.data.train.lr_file.endswith(("limit0.1.txt", "limit0.2.txt"))
        assert cfg.model.get("ste_lift_loss_weight", 0) == 0
        assert "_delete_" not in repr(dict(cfg.model))
        assert "/Users/" not in cfg.pretty_text and "/root/" not in cfg.pretty_text
        # Verify all transform types exist, without importing CUDA backbones.
        from pointcept.datasets.transform import TRANSFORMS

        for split in ("train", "val"):
            for transform in cfg.data[split].transform:
                assert transform["type"] in TRANSFORMS.module_dict


def test_probability_calibration_preserves_rank_and_expected_budget():
    logits = np.linspace(-8, 8, 101)
    prob = calibrate_probability_to_target_count(logits, 23)
    assert abs(prob.sum() - 23) < 1e-8
    assert np.all(np.diff(prob) > 0)
    assert np.all(calibrate_probability_to_target_count(logits, 0) == 0)
    assert np.all(calibrate_probability_to_target_count(logits, 101) == 1)
    samples = [
        len(
            sample_predicted_probability(
                prob, mode="bernoulli", scene_key="toy", seed=i
            )
        )
        for i in range(500)
    ]
    assert abs(np.mean(samples) - 23) < 0.5
    a = sample_predicted_probability(prob, mode="bernoulli", scene_key="toy", seed=1)
    b = sample_predicted_probability(prob, mode="bernoulli", scene_key="toy", seed=1)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scheme", ["grid", "oracle", "retrain"])
def test_scannet_validation_keeps_full_cloud_with_separate_selection(tmp_path, scheme):
    from pointcept.datasets import build_dataset

    cfg = Config.fromfile(str(ROOT / f"configs/scannet/{scheme}.py"))
    assert cfg.data.val.data_root == cfg.data.test.data_root == "data/scannet"
    if scheme != "grid":
        assert cfg.model.eval_map_to_full is True
    scene = tmp_path / "val" / "scene0000_00"
    scene.mkdir(parents=True)
    # Close points catch accidental GridSample as well as selection/cropping.
    coord = np.array([[0, 0, 0], [0.001, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], np.float32)
    labels = np.array([0, 1, 2, -1], np.int32)
    for name, value in {
        "coord": coord,
        "color": np.zeros((4, 3), np.float32),
        "normal": np.ones((4, 3), np.float32),
        "segment20": labels,
    }.items():
        np.save(scene / f"{name}.npy", value)
    indices = tmp_path / "indices.pth"
    torch.save({"scene0000_00": torch.tensor([0, 2])}, indices)
    cfg.data.val.data_root = str(tmp_path)
    if scheme != "grid":
        cfg.data.val.selected_index_file = str(indices)
    item = build_dataset(cfg.data.val)[0]
    if scheme == "grid":
        assert len(item["coord"]) < len(coord)
        assert len(item["inverse"]) == len(coord)
        full_coord, full_segment = item["origin_coord"], item["origin_segment"]
    else:
        full_coord, full_segment = item["coord"], item["segment"]
        torch.testing.assert_close(item["sampled_index"], torch.tensor([0, 2]))
    assert len(full_coord) == len(coord)
    torch.testing.assert_close(full_segment, torch.tensor(labels).long())
    # CenterShift may translate coordinates, but must preserve the full geometry.
    torch.testing.assert_close(
        full_coord - full_coord[0], torch.from_numpy(coord - coord[0])
    )


def test_scannet_grid_uses_paper_online_sampling_and_schedule():
    cfg = Config.fromfile(str(ROOT / "configs/scannet/grid.py"))
    assert cfg.model.type == "DefaultSegmentorV2"
    assert cfg.optimizer.lr == 0.012
    assert cfg.scheduler.max_lr == [0.012, 0.0012]
    assert cfg.param_dicts == [dict(keyword="block", lr=0.0012)]
    assert (cfg.batch_size, cfg.epoch, cfg.eval_epoch, cfg.seed) == (24, 800, 10, 433766)
    transforms = cfg.data.train.transform
    names = [t["type"] for t in transforms]
    assert "ApplySampledIndex" not in names
    assert names.index("RandomScale") < names.index("GridSample")
    for split in ("train", "val"):
        assert not cfg.data[split].get("selected_index_file")
        grid = next(t for t in cfg.data[split].transform if t["type"] == "GridSample")
        assert grid["grid_size"] == 0.04


@pytest.mark.parametrize(
    "stage,batch,loops,seed,lr",
    [
        ("oracle", 24, 800, 433766, 0.006),
        ("sampler", 12, 2000, 433774, 0.003),
        ("retrain", 24, 1600, 433766, 0.006),
        ("grid", 24, 800, 433766, 0.012),
    ],
)
def test_scannet_default_training_recipe(stage, batch, loops, seed, lr):
    cfg = Config.fromfile(str(ROOT / f"configs/scannet/{stage}.py"))
    assert (cfg.batch_size, cfg.epoch, cfg.seed, cfg.optimizer.lr) == (
        batch, loops, seed, lr
    )
    assert cfg.num_worker == (16 if stage == "grid" else 8)
    assert cfg.data.train.lr_file == "configs/scannet/limit0.1.txt"
    scene_list = (ROOT / cfg.data.train.lr_file).read_bytes()
    assert len(set(scene_list.decode().splitlines())) == 120
    assert hashlib.sha256(scene_list).hexdigest() == (
        "c9fa2e94a409ecd2252e8ac507a66aa829e1692b0268a031dfdaeac031f88006"
    )
    assert cfg.enable_amp and cfg.amp_dtype == "float16"
    assert cfg.gradient_accumulation_steps == 1


def test_prepare_outputs_load_into_dataset(tmp_path):
    data = tmp_path / "data"
    for split in ("train", "val"):
        p = data / split / "scene0000_00"
        p.mkdir(parents=True)
        np.save(
            p / "coord.npy",
            np.array([[0, 0, 0], [0.01, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], np.float32),
        )
        np.save(p / "color.npy", np.zeros((4, 3), np.float32))
        np.save(p / "normal.npy", np.ones((4, 3), np.float32))
        np.save(p / "segment20.npy", np.array([0, 0, -1, -1], np.int32))
    cfg = tmp_path / "config.py"
    cfg.write_text(
        "oracle_include_ignore_as_class = True\n"
        "oracle_final_grid_cleanup_size = 0.02\n"
        "oracle_search_cleanup_budget = True\ndata = "
        + repr(
            dict(
                num_classes=20,
                ignore_index=-1,
                **{
                    split: dict(
                        type="ScanNetDataset",
                        split=split,
                        data_root=str(data),
                        transform=[],
                        test_mode=False,
                    )
                    for split in ("train", "val")
                },
            )
        )
    )
    from pointcept.datasets import build_dataset

    for kind in ("grid", "oracle", "probabilities"):
        out = tmp_path / kind
        prepare(str(cfg), out, kind, 0.1, 0.02, min_points=1)
        policy = json.loads((out / f"{kind}_preparation.json").read_text())["policy"]
        assert policy["include_ignore_as_class"]
        assert policy["final_grid_cleanup_size"] == (0.02 if kind == "oracle" else None)
        assert policy["search_cleanup_budget"] == (kind == "oracle")
        field = (
            "marginal_prob_file" if kind == "probabilities" else "selected_index_file"
        )
        suffix = "probabilities" if kind == "probabilities" else "indices"
        ds = build_dataset(
            dict(
                type="ScanNetDataset",
                split="train",
                data_root=str(data),
                transform=[],
                test_mode=False,
                **{field: str(out / f"train_{suffix}.pth")},
            )
        )
        item = ds.get_data(0)
        if kind == "probabilities":
            assert len(item["marginal_prob"]) == 4
            assert np.isfinite(item["marginal_prob"]).all()
            assert item["marginal_prob"][2:].sum() > 0
        else:
            assert 0 < len(item["sampled_index"]) <= 4
            assert np.all(item["sampled_index"] < 4)


def test_mse_matches_paper_and_handles_ignored_points():
    from pointcept.models.selectors.marginal import marginal_probability_mse_loss

    pred = torch.tensor([0.1, 0.8, 0.6], requires_grad=True)
    target = torch.tensor([0.0, 1.0, float("nan")])
    loss = marginal_probability_mse_loss(
        pred, target, valid_mask=torch.isfinite(target)
    )
    torch.testing.assert_close(loss, torch.tensor(0.025))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([0.1, -0.2, 0.0]))
    assert (
        marginal_probability_mse_loss(
            pred, target, valid_mask=torch.zeros(3, dtype=torch.bool)
        ).item()
        == 0
    )


def test_fixed_selector_keeps_batch_indices_separate():
    from pointcept.models.selectors.fixed import FixedSelector

    selector = FixedSelector()
    data = dict(
        coord=torch.arange(15).reshape(5, 3).float(),
        feat=torch.ones(5, 2),
        segment=torch.arange(5),
        offset=torch.tensor([2, 5]),
        sampled_index=torch.tensor([1, 0, 2]),
        sampled_index_offset=torch.tensor([1, 3]),
    )
    output = selector(data)
    torch.testing.assert_close(output.selected_index, torch.tensor([1, 2, 4]))
    torch.testing.assert_close(output.sampled_input["offset"], torch.tensor([1, 3]))
    torch.testing.assert_close(output.sampled_input["segment"], torch.tensor([1, 2, 4]))


@pytest.mark.gpu
def test_cuda_model_construction_and_sampler_backward():
    if not torch.cuda.is_available():
        pytest.skip("Requires NVIDIA CUDA, spconv, torch-scatter and FlashAttention")
    from pointcept.models import build_model

    cfg = Config.fromfile(str(ROOT / "configs/scannet/sampler.py"))
    model = build_model(cfg.model).cuda().train()
    data = dict(
        coord=torch.rand(512, 3, device="cuda"),
        feat=torch.rand(512, 6, device="cuda"),
        grid_size=0.02,
        offset=torch.tensor([512], device="cuda"),
        segment=torch.zeros(512, dtype=torch.long, device="cuda"),
        marginal_prob=torch.rand(512, device="cuda"),
    )
    loss = model(data)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_clean_sampler_forward_backward_and_label_independent_prediction():
    from pointcept.models.builder import MODELS
    from pointcept.models.selectors.marginal import MarginalSamplerRegressor

    @MODELS.register_module(name="TestFeatureBackbone", force=True)
    class FeatureBackbone(torch.nn.Module):
        def forward(self, data):
            return data["feat"]

    model = MarginalSamplerRegressor(
        sampler_type="generic",
        sampler_backbone=dict(type="TestFeatureBackbone"),
        sampler_backbone_out_channels=3,
        sampler_hidden_channels=4,
    )
    data = dict(
        feat=torch.randn(8, 3), segment=torch.arange(8), marginal_prob=torch.rand(8)
    )
    expected = ((model.predict_proba(data) - data["marginal_prob"]) ** 2).mean()
    actual = model(data)["loss"]
    torch.testing.assert_close(actual, expected)
    before = model.predict_proba(data)
    data["segment"] = torch.full((8,), -1)
    torch.testing.assert_close(before, model.predict_proba(data))
    actual.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
