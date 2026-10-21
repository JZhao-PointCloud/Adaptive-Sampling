from pathlib import Path
import numpy as np
from tools import precompute_fixed_selector_indices as precompute


def test_load_scene_auto_supports_s3dis_segment(tmp_path):
    scene_dir = Path(tmp_path) / "Area_1" / "office_0"
    scene_dir.mkdir(parents=True)
    np.save(scene_dir / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(scene_dir / "segment.npy", np.array([3, 4], dtype=np.int32))
    coord, segment = precompute.load_scene(scene_dir)
    assert coord.shape == (2, 3)
    assert np.array_equal(segment, np.array([3, 4], dtype=np.int64))


def test_load_scene_auto_supports_partnet_segment_level_max(tmp_path):
    scene_dir = Path(tmp_path) / "train" / "chair_0000"
    scene_dir.mkdir(parents=True)
    np.save(scene_dir / "coord.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(scene_dir / "segment_level_1.npy", np.array([0, 1], dtype=np.int32))
    np.save(scene_dir / "segment_level_3.npy", np.array([2, 3], dtype=np.int32))
    _, segment = precompute.load_scene(scene_dir)
    assert np.array_equal(segment, np.array([2, 3], dtype=np.int64))


def test_resolve_scene_dirs_uses_relative_s3dis_scene_list(tmp_path):
    root = Path(tmp_path)
    (root / "Area_1" / "office_0").mkdir(parents=True)
    (root / "Area_2" / "office_1").mkdir(parents=True)
    scene_list = root / "limit0.1.txt"
    scene_list.write_text("Area_1/office_0\nArea_2/office_1\n")
    scene_dirs = precompute.resolve_scene_dirs(root, "train", scene_list)
    assert scene_dirs == [
        ("Area_1-office_0", root / "Area_1" / "office_0"),
        ("Area_2-office_1", root / "Area_2" / "office_1"),
    ]
