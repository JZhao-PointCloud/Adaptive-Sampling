_base_ = ["./_segmentor.py"]
save_path = "outputs/s3dis/retrain"
data = {
    "train": {"selected_index_file": "artifacts/s3dis/learned/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/s3dis/learned/val_indices.pth"},
}
