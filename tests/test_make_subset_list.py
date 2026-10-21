import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "make_subset_list.py"
spec = importlib.util.spec_from_file_location("make_subset_list", SCRIPT)
make_subset_list = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_subset_list)


def test_make_subset_list_samples_each_split_deterministically(tmp_path):
    root = tmp_path / "data"
    for split, count in [("Area_1", 5), ("Area_2", 4)]:
        for idx in range(count):
            (root / split / f"room_{idx}").mkdir(parents=True)
    split_entries = make_subset_list.collect_split_entries(root, ["Area_1", "Area_2"])
    selected_a, summary_a = make_subset_list.sample_entries(
        split_entries, ratio=0.4, seed=7, min_count=1
    )
    selected_b, summary_b = make_subset_list.sample_entries(
        split_entries, ratio=0.4, seed=7, min_count=1
    )
    assert selected_a == selected_b
    assert summary_a == summary_b
    assert summary_a == [("Area_1", 5, 2), ("Area_2", 4, 2)]
    assert len([entry for entry in selected_a if entry.startswith("Area_1/")]) == 2
    assert len([entry for entry in selected_a if entry.startswith("Area_2/")]) == 2


def test_make_subset_list_writes_relative_entries(tmp_path):
    root = tmp_path / "data"
    for name in ["chair_0000", "chair_0001", "table_0000"]:
        (root / "train" / name).mkdir(parents=True)
    split_entries = make_subset_list.collect_split_entries(root, ["train"])
    selected, _ = make_subset_list.sample_entries(
        split_entries, ratio=1.0, seed=0, min_count=1
    )
    output = tmp_path / "limit0.1.txt"
    make_subset_list.write_entries(selected, output)
    assert output.read_text().splitlines() == [
        "train/chair_0000",
        "train/chair_0001",
        "train/table_0000",
    ]
