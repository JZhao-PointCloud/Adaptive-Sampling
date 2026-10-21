_base_ = ["./_segmentor.py"]
save_path = "outputs/scannet/oracle"
oracle_include_ignore_as_class = True
oracle_final_grid_cleanup_size = 0.02
oracle_search_cleanup_budget = True
data = {
    "train": {"selected_index_file": "artifacts/scannet/oracle/train_indices.pth"},
    "val": {"selected_index_file": "artifacts/scannet/oracle/val_indices.pth"},
}
