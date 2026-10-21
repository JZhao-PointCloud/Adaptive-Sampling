from pathlib import Path
import numpy as np
import torch
from pointcept.datasets.partnet import PartNetDataset
from pointcept.datasets.s3dis import S3DISDataset


def _write_sample(root, split, name, labels):
    sample_dir = root / split / name
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coord.npy", np.zeros((len(labels), 3), dtype=np.float32))
    np.save(sample_dir / "segment_level_1.npy", np.array(labels, dtype=np.int32))


def test_partnet_dataset_uses_lr_file_subset(tmp_path):
    root = Path(tmp_path)
    _write_sample(root, "train", "chair_0000", [0, 1])
    _write_sample(root, "train", "chair_0001", [0, 0])
    _write_sample(root, "train", "table_0000", [0])
    lr_file = root / "limit0.1.txt"
    lr_file.write_text("train/chair_0001\n")
    dataset = PartNetDataset(
        split="train",
        data_root=str(root),
        segment_level=1,
        lr_file=str(lr_file),
        transform=[],
        test_mode=False,
    )
    assert len(dataset.data_list) == 1
    assert dataset.data_list[0] == str(root / "train" / "chair_0001")
    data = dataset.get_data(0)
    assert data["name"] == "chair_0001"
    assert data["split"] == "train"


def test_partnet_dataset_loads_marginal_probability(tmp_path):
    root = Path(tmp_path)
    _write_sample(root, "train", "chair_0000", [0, 1])
    marginal_prob_file = root / "marginal_prob.pth"
    torch.save({"train/chair_0000": torch.tensor([0.25, 0.75])}, marginal_prob_file)
    dataset = PartNetDataset(
        split="train",
        data_root=str(root),
        segment_level=1,
        marginal_prob_file=str(marginal_prob_file),
        transform=[],
        test_mode=False,
    )
    data = dataset.get_data(0)
    assert np.allclose(data["marginal_prob"], np.array([0.25, 0.75], dtype=np.float32))


def test_partnet_dataset_loads_selected_index(tmp_path):
    root = Path(tmp_path)
    _write_sample(root, "train", "chair_0000", [0, 1, 2])
    selected_index_file = root / "selected_index.pth"
    torch.save({"chair_0000": torch.tensor([0, 2])}, selected_index_file)
    dataset = PartNetDataset(
        split="train",
        data_root=str(root),
        segment_level=1,
        selected_index_file=str(selected_index_file),
        transform=[],
        test_mode=False,
    )
    data = dataset.get_data(0)
    assert np.array_equal(data["sampled_index"], np.array([0, 2], dtype=np.int64))


def test_s3dis_dataset_loads_marginal_probability_by_area_room_key(tmp_path):
    root = Path(tmp_path)
    sample_dir = root / "Area_1" / "office_0"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coord.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "color.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "normal.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "segment.npy", np.array([0, 1, 2], dtype=np.int32))
    np.save(sample_dir / "instance.npy", np.array([0, 0, 1], dtype=np.int32))
    marginal_prob_file = root / "marginal_prob.pth"
    torch.save({"Area_1-office_0": torch.tensor([0.1, 0.2, 0.3])}, marginal_prob_file)
    dataset = S3DISDataset(
        split="Area_1",
        data_root=str(root),
        marginal_prob_file=str(marginal_prob_file),
        transform=[],
        test_mode=False,
    )
    data = dataset.get_data(0)
    assert data["name"] == "Area_1-office_0"
    assert np.allclose(
        data["marginal_prob"], np.array([0.1, 0.2, 0.3], dtype=np.float32)
    )


def test_s3dis_dataset_loads_selected_index_by_relative_key(tmp_path):
    root = Path(tmp_path)
    sample_dir = root / "Area_1" / "office_0"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "coord.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "color.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "normal.npy", np.zeros((3, 3), dtype=np.float32))
    np.save(sample_dir / "segment.npy", np.array([0, 1, 2], dtype=np.int32))
    np.save(sample_dir / "instance.npy", np.array([0, 0, 1], dtype=np.int32))
    selected_index_file = root / "selected_index.pth"
    torch.save({"Area_1/office_0": torch.tensor([0, 2])}, selected_index_file)
    dataset = S3DISDataset(
        split="Area_1",
        data_root=str(root),
        selected_index_file=str(selected_index_file),
        transform=[],
        test_mode=False,
    )
    data = dataset.get_data(0)
    assert np.array_equal(data["sampled_index"], np.array([0, 2], dtype=np.int64))
