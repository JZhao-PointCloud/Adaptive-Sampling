_base_ = ["./_segmentor.py"]
save_path = "outputs/scannet/retrain"
epoch = 1600
data = {
    "train": {"selected_index_file": "artifacts/scannet/learned/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/scannet/learned/val_indices.pth"},
}
