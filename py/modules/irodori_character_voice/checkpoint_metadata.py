from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_META_KEY = "config_json"
LEGACY_MODEL_CONFIG_META_KEY = "model_config_json"
LEGACY_INFERENCE_CONFIG_META_KEY = "inference_config_json"

INFERENCE_CONFIG_KEYS = ("max_text_len", "max_caption_len", "fixed_target_latent_steps")

_BASE_MODEL_CONFIG_KEYS = (
    "latent_dim",
    "latent_patch_size",
    "model_dim",
    "num_layers",
    "num_heads",
    "mlp_ratio",
    "text_mlp_ratio",
)

_BASE_TEXT_MODEL_CONFIG_KEYS = (
    "dropout",
    "text_vocab_size",
    "text_tokenizer_repo",
    "text_add_bos",
    "text_dim",
    "text_layers",
    "text_heads",
)

_SPEAKER_RATIO_MODEL_CONFIG_KEYS = ("speaker_mlp_ratio",)

_SPEAKER_MODEL_CONFIG_KEYS = (
    "speaker_dim",
    "speaker_layers",
    "speaker_heads",
    "speaker_patch_size",
)

_CAPTION_MODEL_CONFIG_KEYS = (
    "use_caption_condition",
    "caption_vocab_size",
    "caption_tokenizer_repo",
    "caption_add_bos",
    "caption_dim",
    "caption_layers",
    "caption_heads",
    "caption_mlp_ratio",
)

_CHARACTER_MODEL_CONFIG_KEYS = (
    "use_character_condition",
    "character_encoder_model",
    "character_dim",
    "character_use_all_patches",
    "character_image_size",
    "character_hidden_state_index",
    "character_projector",
    "character_attention_mode",
)

_TAIL_MODEL_CONFIG_KEYS = (
    "timestep_embed_dim",
    "adaln_rank",
    "norm_eps",
)


def _path_label(path: Path | None) -> str:
    return "" if path is None else f": {path}"


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}'{_path_label(path)}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata{_path_label(path)}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object{_path_label(path)}")
    return payload


def _copy_present(
    source: Mapping[str, Any],
    target: dict[str, Any],
    keys: tuple[str, ...],
    *,
    skip_none: bool = True,
) -> None:
    for key in keys:
        if key not in source:
            continue
        value = source[key]
        if value is None and skip_none:
            continue
        target[key] = value


def extract_inference_config(raw: Mapping[str, Any] | None) -> dict[str, int]:
    if raw is None:
        return {}

    inference_cfg: dict[str, int] = {}
    for key in INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)
    return inference_cfg


def build_model_metadata_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    use_caption = bool(model_config.get("use_caption_condition", False))
    use_character = bool(model_config.get("use_character_condition", False))
    use_speaker = not use_caption and not use_character

    _copy_present(model_config, cfg, _BASE_MODEL_CONFIG_KEYS)
    if use_speaker:
        _copy_present(model_config, cfg, _SPEAKER_RATIO_MODEL_CONFIG_KEYS)
    _copy_present(model_config, cfg, _BASE_TEXT_MODEL_CONFIG_KEYS)
    if use_caption:
        _copy_present(model_config, cfg, _CAPTION_MODEL_CONFIG_KEYS)
    if use_character:
        _copy_present(model_config, cfg, _CHARACTER_MODEL_CONFIG_KEYS)
    if use_speaker:
        _copy_present(model_config, cfg, _SPEAKER_MODEL_CONFIG_KEYS)
    _copy_present(model_config, cfg, _TAIL_MODEL_CONFIG_KEYS)

    return cfg


def build_flat_safetensors_config(
    *,
    model_config: Mapping[str, Any],
    inference_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    flat_config = build_model_metadata_config(model_config)
    flat_config.update(extract_inference_config(inference_config))
    return flat_config


def build_safetensors_metadata(
    *,
    model_config: Mapping[str, Any] | None = None,
    inference_config: Mapping[str, Any] | None = None,
    flat_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if flat_config is None:
        if model_config is None:
            raise ValueError("model_config is required when flat_config is not provided.")
        flat_payload = build_flat_safetensors_config(
            model_config=model_config,
            inference_config=inference_config,
        )
    else:
        flat_payload = dict(flat_config)

    return {
        CONFIG_META_KEY: json.dumps(flat_payload, ensure_ascii=False, separators=(",", ":")),
    }


def split_flat_safetensors_config(
    flat_config: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    model_cfg: dict[str, Any] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in INFERENCE_CONFIG_KEYS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata"
                    f"{_path_label(path)}"
                )
            inference_cfg[key] = int(value)
        else:
            model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def read_safetensors_metadata_config(
    metadata: Mapping[str, str] | None,
    *,
    path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    metadata = metadata or {}
    flat_config = _parse_json_mapping(
        metadata.get(CONFIG_META_KEY),
        field=CONFIG_META_KEY,
        path=path,
    )
    if flat_config is not None:
        return split_flat_safetensors_config(flat_config, path=path)

    model_cfg = _parse_json_mapping(
        metadata.get(LEGACY_MODEL_CONFIG_META_KEY),
        field=LEGACY_MODEL_CONFIG_META_KEY,
        path=path,
    )
    inference_cfg = _parse_json_mapping(
        metadata.get(LEGACY_INFERENCE_CONFIG_META_KEY),
        field=LEGACY_INFERENCE_CONFIG_META_KEY,
        path=path,
    )
    return dict(model_cfg or {}), (extract_inference_config(inference_cfg) or None)
