_base_ = ["./_segmentor.py"]
save_path = "outputs/partnet/retrain"
data = {
    "train": {"selected_index_file": "artifacts/partnet/learned/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/partnet/learned/val_indices.pth"},
}
