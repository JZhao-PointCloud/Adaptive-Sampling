_base_ = ["./_segmentor.py"]
save_path = "outputs/s3dis/grid"
data = {
    "train": {"selected_index_file": "artifacts/s3dis/grid/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/s3dis/grid/val_indices.pth"},
}
