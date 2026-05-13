from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from .mlp import MLPProjector, MLPProjectorConfig
from .resampler import ResamplerProjector, ResamplerProjectorConfig

ProjectorConfig = MLPProjectorConfig | ResamplerProjectorConfig

_CONFIG_TYPES: dict[str, type] = {
    "mlp": MLPProjectorConfig,
    "resampler": ResamplerProjectorConfig,
}


def resolve_projector_config(payload: dict[str, Any] | ProjectorConfig | None) -> ProjectorConfig:
    """Build a concrete projector config dataclass from a raw mapping."""
    if isinstance(payload, MLPProjectorConfig | ResamplerProjectorConfig):
        return payload
    if payload is None:
        return MLPProjectorConfig()
    if not isinstance(payload, dict):
        raise ValueError(f"projector config must be a mapping, got {type(payload)!r}")

    data = dict(payload)
    proj_type = data.pop("type", "mlp")
    if proj_type not in _CONFIG_TYPES:
        raise ValueError(f"Unknown projector type {proj_type!r}. Known: {sorted(_CONFIG_TYPES)}")

    return _CONFIG_TYPES[proj_type](type=proj_type, **data)


def projector_config_to_dict(config: ProjectorConfig) -> dict[str, Any]:
    return asdict(config)


def _projector_state_items(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix) and isinstance(value, torch.Tensor)
    }


def infer_projector_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "character_encoder.proj.",
    fallback: dict[str, Any] | ProjectorConfig | None = None,
) -> tuple[ProjectorConfig | None, int | None, int | None, list[str]]:
    """
    Infer a character projector config from checkpoint tensor names/shapes.

    Returns:
        (config, backbone_dim, output_dim, warnings)
    """
    warnings: list[str] = []
    state = _projector_state_items(state_dict, prefix)
    if not state:
        return None, None, None, warnings

    resolved_fallback = resolve_projector_config(fallback)

    if "queries" in state or "proj_in.weight" in state:
        proj_in = state.get("proj_in.weight")
        queries = state.get("queries")
        proj_out = state.get("proj_out.weight")
        if proj_in is None or proj_in.ndim != 2:
            raise ValueError("Could not infer resampler projector: missing proj_in.weight.")

        out_dim, in_dim = int(proj_in.shape[0]), int(proj_in.shape[1])
        num_query_tokens = (
            int(queries.shape[1])
            if queries is not None and queries.ndim == 3
            else int(getattr(resolved_fallback, "num_query_tokens", 8))
        )
        if proj_out is not None and proj_out.ndim == 2:
            out_dim = int(proj_out.shape[0])

        depth = 0
        for key in state:
            if key.startswith("blocks."):
                try:
                    depth = max(depth, int(key.split(".", 2)[1]) + 1)
                except (IndexError, ValueError):
                    pass
        if depth <= 0:
            depth = int(getattr(resolved_fallback, "depth", 4))

        w1 = state.get("blocks.0.mlp.w1.weight")
        mlp_ratio = (
            float(w1.shape[0]) / float(out_dim)
            if w1 is not None and w1.ndim == 2 and out_dim > 0
            else float(getattr(resolved_fallback, "mlp_ratio", 2.0))
        )

        qk_norm = any(key.endswith(".attn.norm_q.weight") for key in state)
        is_gated = any(key.endswith(".attn.to_gate.weight") for key in state)
        norm_q = state.get("blocks.0.attn.norm_q.weight")
        if norm_q is not None and norm_q.ndim == 1 and int(norm_q.shape[0]) > 0:
            num_heads = max(1, out_dim // int(norm_q.shape[0]))
        else:
            num_heads = int(getattr(resolved_fallback, "num_heads", 8))
            warnings.append(
                "warning: resampler num_heads could not be inferred from state_dict; "
                f"using {num_heads}."
            )

        config = ResamplerProjectorConfig(
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_query_tokens=num_query_tokens,
            depth=depth,
            gradient_checkpointing=bool(getattr(resolved_fallback, "gradient_checkpointing", False)),
            qk_norm=qk_norm,
            is_gated=is_gated,
        )
        return config, in_dim, out_dim, warnings

    fc1_keys = sorted(
        key for key in state if key.startswith("blocks.") and key.endswith(".fc1.weight")
    )
    fc2_keys = sorted(
        key for key in state if key.startswith("blocks.") and key.endswith(".fc2.weight")
    )
    if fc1_keys and fc2_keys:
        num_layers = 0
        for key in fc1_keys:
            try:
                num_layers = max(num_layers, int(key.split(".", 2)[1]) + 1)
            except (IndexError, ValueError):
                pass
        first_fc1 = state[fc1_keys[0]]
        last_fc2 = state[fc2_keys[-1]]
        if first_fc1.ndim != 2 or last_fc2.ndim != 2:
            raise ValueError("Could not infer MLP projector: fc weights must be 2D.")
        hidden_dim = int(first_fc1.shape[0])
        in_dim = int(first_fc1.shape[1])
        out_dim = int(last_fc2.shape[0])
        config = MLPProjectorConfig(
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers),
        )
        return config, in_dim, out_dim, warnings

    warnings.append("warning: character projector state_dict keys were present but unrecognized.")
    return None, None, None, warnings


def build_projector(
    config: ProjectorConfig,
    backbone_dim: int,
    output_dim: int,
) -> nn.Module:
    """Factory for building a projector module from a resolved config."""
    if isinstance(config, MLPProjectorConfig):
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else backbone_dim
        return MLPProjector(
            in_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=config.num_layers,
        )
    if isinstance(config, ResamplerProjectorConfig):
        return ResamplerProjector(
            in_dim=backbone_dim,
            out_dim=output_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_query_tokens=config.num_query_tokens,
            depth=config.depth,
            gradient_checkpointing=config.gradient_checkpointing,
            qk_norm=config.qk_norm,
            is_gated=config.is_gated,
        )
    raise ValueError(f"Unknown projector config type: {type(config).__name__}")


__all__ = [
    "MLPProjector",
    "MLPProjectorConfig",
    "ResamplerProjector",
    "ResamplerProjectorConfig",
    "ProjectorConfig",
    "build_projector",
    "infer_projector_config_from_state_dict",
    "resolve_projector_config",
    "projector_config_to_dict",
]
