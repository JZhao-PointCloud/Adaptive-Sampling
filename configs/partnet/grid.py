_base_ = ["./_segmentor.py"]
save_path = "outputs/partnet/grid"
data = {
    "train": {"selected_index_file": "artifacts/partnet/grid/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/partnet/grid/val_indices.pth"},
}
