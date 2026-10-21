_base_ = ["./sampler.py"]

# Used for fixed-index generation and multimask inference.
data = dict(
    val=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="AddPointIndex", key="point_index"),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="Update", keys_dict=dict(grid_size=0.02)),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord", "grid_coord", "grid_size", "segment",
                    "marginal_prob", "point_index",
                ),
                feat_keys=("color", "normal"),
            ),
        ],
    ),
)
