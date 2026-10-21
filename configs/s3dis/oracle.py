_base_ = ["./_segmentor.py"]
save_path = "outputs/s3dis/oracle"
data = {
    "train": {"selected_index_file": "artifacts/s3dis/oracle/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/s3dis/oracle/val_indices.pth"},
}
