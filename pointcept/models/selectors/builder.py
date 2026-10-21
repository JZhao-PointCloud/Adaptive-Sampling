import copy

from pointcept.utils.registry import Registry


SELECTORS = Registry("selectors")


def build_selector(cfg):
    if cfg is None:
        return None
    return SELECTORS.build(copy.deepcopy(cfg))
