"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import importlib
from pointcept.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    if cfg["type"] not in MODELS.module_dict:
        for name in (
            "default",
            "point_transformer_v3",
            "sparse_unet",
            "selectors",
            "selector_segmentor",
        ):
            importlib.import_module(f"pointcept.models.{name}")
    return MODELS.build(copy.deepcopy(cfg))
