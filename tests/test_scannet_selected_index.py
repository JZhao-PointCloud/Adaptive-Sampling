import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch
from pointcept.datasets.scannet import ScanNetDataset
from pointcept.datasets.transform import (
    ApplySampledIndex,
    GridSample,
    ShufflePoint,
    SphereCrop,
)


class TestScanNetSelectedIndex(unittest.TestCase):
    def test_apply_sampled_index_subsets_point_aligned_arrays(self):
        transform = ApplySampledIndex(reset_sampled_index=True)
        data = dict(
            coord=np.arange(12, dtype=np.float32).reshape(4, 3),
            color=np.arange(12, dtype=np.float32).reshape(4, 3),
            segment=np.array([0, 1, 2, 3], dtype=np.int32),
            marginal_prob=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            sampled_index=np.array([1, 3], dtype=np.int64),
        )
        out = transform(data)
        self.assertEqual(out["coord"].shape[0], 2)
        self.assertTrue(
            np.array_equal(out["segment"], np.array([1, 3], dtype=np.int32))
        )
        self.assertTrue(
            np.allclose(out["marginal_prob"], np.array([0.2, 0.4], dtype=np.float32))
        )
        self.assertTrue(
            np.array_equal(out["sampled_index"], np.array([0, 1], dtype=np.int64))
        )

    def test_scannet_dataset_loads_selected_index_without_masking_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "train" / "scene0000_00"
            scene_dir.mkdir(parents=True)
            np.save(scene_dir / "coord.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "color.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "normal.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "segment20.npy", np.array([0, 1, 2, 3], dtype=np.int32))
            selected_index_file = root / "selected_index.pth"
            torch.save(
                {"scene0000_00": torch.tensor([1, 3], dtype=torch.long)},
                selected_index_file,
            )
            dataset = ScanNetDataset(
                split="train",
                data_root=str(root),
                transform=[],
                test_mode=False,
                selected_index_file=str(selected_index_file),
            )
            data = dataset.get_data(0)
            self.assertTrue(
                np.array_equal(data["sampled_index"], np.array([1, 3], dtype=np.int64))
            )
            self.assertTrue(
                np.array_equal(data["segment"], np.array([0, 1, 2, 3], dtype=np.int32))
            )

    def test_scannet_dataset_loads_marginal_probability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_dir = root / "train" / "scene0000_00"
            scene_dir.mkdir(parents=True)
            np.save(scene_dir / "coord.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "color.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "normal.npy", np.zeros((4, 3), dtype=np.float32))
            np.save(scene_dir / "segment20.npy", np.array([0, 1, 2, 3], dtype=np.int32))
            marginal_prob_file = root / "marginal_prob.pth"
            torch.save(
                {"scene0000_00": torch.tensor([0.1, 0.2, 0.3, 0.4])}, marginal_prob_file
            )
            dataset = ScanNetDataset(
                split="train",
                data_root=str(root),
                transform=[],
                test_mode=False,
                marginal_prob_file=str(marginal_prob_file),
            )
            data = dataset.get_data(0)
            self.assertTrue(
                np.allclose(
                    data["marginal_prob"],
                    np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                )
            )

    def test_sphere_crop_preserves_and_remaps_sampled_index(self):
        transform = SphereCrop(point_max=3, mode="center")
        data = dict(
            coord=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            color=np.arange(18, dtype=np.float32).reshape(6, 3),
            segment=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            sampled_index=np.array([4, 5], dtype=np.int64),
        )
        out = transform(data)
        self.assertGreaterEqual(out["coord"].shape[0], 5)
        self.assertTrue(
            np.array_equal(
                out["coord"][out["sampled_index"]],
                np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                out["segment"][out["sampled_index"]], np.array([4, 5], dtype=np.int32)
            )
        )

    def test_shuffle_point_remaps_sampled_index(self):
        transform = ShufflePoint()
        data = dict(
            coord=np.arange(18, dtype=np.float32).reshape(6, 3),
            color=np.arange(18, dtype=np.float32).reshape(6, 3),
            segment=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            sampled_index=np.array([1, 4], dtype=np.int64),
        )
        out = transform(data)
        self.assertTrue(
            np.array_equal(
                np.sort(out["segment"][out["sampled_index"]]),
                np.array([1, 4], dtype=np.int32),
            )
        )

    def test_apply_sampled_index_can_mark_fixed_oracle_subset(self):
        transform = ApplySampledIndex(
            reset_sampled_index=True, mark_fixed_oracle_subset=True
        )
        data = dict(
            coord=np.arange(12, dtype=np.float32).reshape(4, 3),
            color=np.arange(12, dtype=np.float32).reshape(4, 3),
            segment=np.array([0, 1, 2, 3], dtype=np.int32),
            sampled_index=np.array([1, 3], dtype=np.int64),
        )
        out = transform(data)
        self.assertTrue(out["fixed_oracle_subset_applied"])
        self.assertTrue(out["skip_train_gridsample_after_apply_sampled_index"])
        self.assertTrue(out["preserve_sampled_index_on_point_drop"])
        self.assertTrue(
            np.array_equal(out["sampled_index"], np.array([0, 1], dtype=np.int64))
        )

    def test_gridsample_train_skips_after_fixed_oracle_subset(self):
        transform = GridSample(grid_size=0.5, mode="train")
        data = dict(
            coord=np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
            color=np.arange(12, dtype=np.float32).reshape(4, 3),
            normal=np.zeros((4, 3), dtype=np.float32),
            segment=np.array([0, 1, 2, 3], dtype=np.int32),
            sampled_index=np.array([0, 1, 2, 3], dtype=np.int64),
            fixed_oracle_subset_applied=True,
            skip_train_gridsample_after_apply_sampled_index=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = transform(data)
        self.assertEqual(out["coord"].shape[0], 4)
        self.assertTrue(
            np.array_equal(out["sampled_index"], np.array([0, 1, 2, 3], dtype=np.int64))
        )
        self.assertEqual(len(caught), 1)
        self.assertIn("Skipping GridSample(train)", str(caught[0].message))

    def test_gridsample_train_runs_when_fixed_oracle_skip_is_disabled(self):
        transform = GridSample(grid_size=0.5, mode="train")
        data = dict(
            coord=np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
            color=np.arange(12, dtype=np.float32).reshape(4, 3),
            normal=np.zeros((4, 3), dtype=np.float32),
            segment=np.array([0, 1, 2, 3], dtype=np.int32),
            sampled_index=np.array([0, 1, 2, 3], dtype=np.int64),
            fixed_oracle_subset_applied=True,
            skip_train_gridsample_after_apply_sampled_index=False,
            preserve_sampled_index_on_point_drop=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = transform(data)
        self.assertEqual(out["coord"].shape[0], 2)
        self.assertEqual(len(caught), 0)

    def test_spherecrop_can_drop_points_when_preserve_sampled_index_is_disabled(self):
        transform = ApplySampledIndex(
            reset_sampled_index=True,
            mark_fixed_oracle_subset=True,
            preserve_sampled_index_on_point_drop=False,
        )
        crop = SphereCrop(point_max=2, mode="center")
        data = dict(
            coord=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            color=np.arange(12, dtype=np.float32).reshape(4, 3),
            segment=np.array([0, 1, 2, 3], dtype=np.int32),
            sampled_index=np.array([0, 1, 2, 3], dtype=np.int64),
        )
        out = transform(data)
        out = crop(out)
        self.assertEqual(out["coord"].shape[0], 2)
        self.assertEqual(out["sampled_index"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
