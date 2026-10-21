_base_ = ["./_segmentor.py"]
save_path = "outputs/partnet/oracle"
data = {
    "train": {"selected_index_file": "artifacts/partnet/oracle/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/partnet/oracle/val_indices.pth"},
}
