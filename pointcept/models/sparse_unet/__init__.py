try:
    from .spconv_unet_v1m1_base import *
except ImportError as exc:  # pragma: no cover - optional sparse backend.
    import warnings

    warnings.warn(f"SpUNet is unavailable because spconv import failed: {exc}")
