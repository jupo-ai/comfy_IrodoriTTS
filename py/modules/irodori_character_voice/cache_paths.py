from __future__ import annotations

from pathlib import Path


def extension_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def image_encoder_cache_root() -> Path:
    root = extension_data_dir() / "image_encoders"
    root.mkdir(parents=True, exist_ok=True)
    return root


def timm_model_cache_dir() -> Path:
    cache_dir = image_encoder_cache_root()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
