from collections.abc import Callable
from os import PathLike
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image as PILImage
from timm import create_model
from timm import data as timm_data

from ..download_progress import download_hf_file_with_progress, download_log, repo_cache_dir
from .cache_paths import timm_model_cache_dir
from .projector import ProjectorConfig, build_projector, resolve_projector_config


def load_character_image(
    path: str | PathLike[str],
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> PILImage.Image:
    """Open an image and return it as RGB, compositing transparent pixels onto a solid background."""
    img = PILImage.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = PILImage.new("RGB", img.size, background)
        canvas.paste(img, mask=img.split()[-1])
        return canvas
    return img.convert("RGB")


def character_tensor_to_pil(
    image: torch.Tensor,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> PILImage.Image:
    """Convert a ComfyUI image tensor [B, H, W, C] or [H, W, C] to RGB PIL."""
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape [B, H, W, C] or [H, W, C], got {tuple(image.shape)}")

    image = image.detach().cpu().float().clamp(0.0, 1.0)
    channels = int(image.shape[-1])
    if channels == 1:
        image = image.repeat(1, 1, 3)
    elif channels == 4:
        rgb = image[..., :3]
        alpha = image[..., 3:4]
        bg = torch.tensor(background, dtype=rgb.dtype).view(1, 1, 3) / 255.0
        image = rgb * alpha + bg * (1.0 - alpha)
    elif channels > 4:
        image = image[..., :3]

    if int(image.shape[-1]) != 3:
        raise ValueError(f"Expected image tensor channels to be 1, 3, or 4, got {channels}")

    array = (image * 255.0).round().to(torch.uint8).numpy()
    return PILImage.fromarray(array, mode="RGB")


def coerce_character_image(
    image: str | PathLike[str] | torch.Tensor,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> PILImage.Image:
    if isinstance(image, torch.Tensor):
        return character_tensor_to_pil(image, background=background)
    return load_character_image(image, background=background)


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm to avoid circular imports with model.py."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms * self.weight).to(x_dtype)


def build_character_transform(timm_model_id: str, image_size: int) -> Callable:
    """
    Build a torchvision transform for preprocessing character reference images.
    Uses the timm model's own recommended preprocessing config.
    The returned transform is a torchvision Compose and is safe to pickle for
    DataLoader multiprocessing workers.
    """
    m = create_model(timm_model_id, pretrained=False, cache_dir=timm_model_cache_dir())
    data_cfg = timm_data.resolve_model_data_config(m)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm_data.create_transform(**data_cfg, is_training=False)
    del m
    return transform


def _hf_repo_id_from_timm_model_id(timm_model_id: str) -> str | None:
    prefix = "hf_hub:"
    if not str(timm_model_id).startswith(prefix):
        return None
    return str(timm_model_id)[len(prefix) :]


def _download_timm_weights_with_progress(timm_model_id: str) -> Path | None:
    repo_id = _hf_repo_id_from_timm_model_id(timm_model_id)
    if repo_id is None:
        return None

    target_dir = repo_cache_dir(timm_model_cache_dir(), repo_id)
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            return download_hf_file_with_progress(
                repo_id,
                filename,
                target_dir,
                item_type="image_encoder",
            )
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                continue
            raise
    raise RuntimeError(f"No supported timm weight file found in Hugging Face repo: {repo_id}")


def _feature_dim_and_format(out: torch.Tensor, timm_model_id: str) -> tuple[int, str]:
    if out.ndim == 2:
        return int(out.shape[-1]), "pooled"
    if out.ndim == 3:
        return int(out.shape[-1]), "seq"
    if out.ndim == 4:
        return int(out.shape[1]), "spatial"
    raise RuntimeError(f"Unexpected backbone output ndim={out.ndim} for {timm_model_id!r}")


def _hidden_state_to_sequence(hidden_state: torch.Tensor) -> torch.Tensor:
    if hidden_state.ndim == 3:
        return hidden_state
    if hidden_state.ndim == 4:
        batch_size, dim, h, w = hidden_state.size()
        return hidden_state.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)
    raise RuntimeError(f"Unexpected intermediate hidden state ndim={hidden_state.ndim}.")


def resolve_character_hidden_state_index(
    timm_model_id: str,
    *,
    image_size: int,
    expected_dim: int,
    preferred_index: int | None,
) -> tuple[int | None, list[str]]:
    """
    Pick a timm intermediate index whose feature dim matches a checkpoint projector.

    ``None`` means the backbone's normal forward output already matches.
    """
    warnings: list[str] = []
    m = create_model(
        timm_model_id,
        pretrained=False,
        global_pool="",
        cache_dir=timm_model_cache_dir(),
    )
    reset_classifier = getattr(m, "reset_classifier", None)
    if callable(reset_classifier):
        reset_classifier(0)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        if preferred_index is None:
            try:
                out = m(dummy)
                dim, _fmt = _feature_dim_and_format(out, timm_model_id)
                if dim == expected_dim:
                    return None, warnings
            except Exception as exc:
                warnings.append(
                    "warning: direct timm backbone probe failed while resolving "
                    f"character hidden state: {exc}"
                )
        if not hasattr(m, "forward_intermediates"):
            raise ValueError(
                f"Backbone {timm_model_id!r} output does not match projector input "
                f"dim={expected_dim}, and forward_intermediates is unavailable."
            )
        intermediates = m.forward_intermediates(  # type: ignore[attr-defined]
            dummy,
            intermediates_only=True,
        )

    num_intermediates = len(intermediates)
    if preferred_index is not None:
        candidates = [preferred_index]
    else:
        preferred = [-5, -4, -3, -2, -1]
        candidates = [
            *[idx for idx in preferred if -num_intermediates <= idx < num_intermediates],
            *range(num_intermediates),
        ]

    seen: set[int] = set()
    matches: list[int] = []
    for idx in candidates:
        normalized = idx if idx >= 0 else num_intermediates + idx
        if normalized in seen or normalized < 0 or normalized >= num_intermediates:
            continue
        seen.add(normalized)
        hidden_state = _hidden_state_to_sequence(intermediates[idx])
        dim = int(hidden_state.shape[-1])
        if dim == expected_dim:
            matches.append(idx)

    if preferred_index is not None:
        if matches:
            return preferred_index, warnings
        raise ValueError(
            f"character_hidden_state_index={preferred_index} does not produce projector "
            f"input dim={expected_dim} for {timm_model_id!r}."
        )
    if matches:
        if len(matches) > 1:
            warnings.append(
                "warning: multiple timm hidden states match the character projector input "
                f"dim={expected_dim}; using {matches[0]}."
            )
        return matches[0], warnings
    raise ValueError(
        f"Could not find a timm hidden state for {timm_model_id!r} with feature dim={expected_dim}. "
        "Pass --character-hidden-state-index explicitly."
    )


class CharacterImageEncoder(nn.Module):
    """
    Character reference image encoder using a timm backbone.

    Encodes a batch of preprocessed images into a sequence of patch-level
    feature vectors projected to ``output_dim``.

    Args:
        timm_model_id: timm model identifier, e.g.
            ``"hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"``.
        output_dim: Projected output dimension fed into JointAttention.
        use_all_patches: If True, use all spatial patch tokens.
            If False, use only the CLS token (index 0).
        image_size: Image resize target (H = W).
        pretrained: Whether to load pretrained backbone weights.
        projector_config: Configuration for the projector module.
    """

    def __init__(
        self,
        timm_model_id: str,
        output_dim: int,
        use_all_patches: bool = True,
        image_size: int = 448,
        pretrained: bool = True,
        projector_config: ProjectorConfig | dict | None = None,
        hidden_state_index: int | None = None,
    ):
        super().__init__()

        if projector_config is None or isinstance(projector_config, dict):
            projector_config = resolve_projector_config(projector_config)

        # Load backbone without classification head, retaining all spatial tokens.
        cache_dir = timm_model_cache_dir()
        if pretrained:
            weights_path = _download_timm_weights_with_progress(timm_model_id)
            if weights_path is None:
                with download_log(
                    "image_encoder",
                    timm_model_id,
                    cache_dir=cache_dir,
                    progress="huggingface",
                ):
                    backbone = create_model(
                        timm_model_id,
                        pretrained=True,
                        global_pool="",
                        cache_dir=cache_dir,
                    )
            else:
                with download_log(
                    "image_encoder",
                    str(weights_path),
                    cache_dir=cache_dir,
                    progress="local",
                ):
                    backbone = create_model(
                        timm_model_id,
                        pretrained=True,
                        global_pool="",
                        cache_dir=cache_dir,
                        pretrained_cfg_overlay={
                            "file": str(weights_path),
                            "source": "",
                        },
                    )
        else:
            backbone = create_model(
                timm_model_id,
                pretrained=False,
                global_pool="",
                cache_dir=cache_dir,
            )
        reset_classifier = getattr(backbone, "reset_classifier", None)
        if callable(reset_classifier):
            reset_classifier(0)  # remove classifier head later
        self.backbone = backbone
        self.use_all_patches = use_all_patches
        self._image_size = image_size
        self._hidden_state_index = hidden_state_index

        if hidden_state_index is not None and not hasattr(backbone, "forward_intermediates"):
            raise ValueError(
                f"Backbone {timm_model_id!r} does not support forward_intermediates; "
                "hidden_state_index cannot be used."
            )

        # Detect backbone output feature dimension via a probe forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            out = self._backbone_forward(dummy)
            backbone_dim, self._out_format = _feature_dim_and_format(out, timm_model_id)

        self.proj = build_projector(projector_config, backbone_dim, output_dim)
        # Post-projection norm for stable training (mirrors caption_norm pattern).
        self.norm = _RMSNorm(output_dim)

    def _backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the backbone, optionally returning an intermediate hidden state."""
        if self._hidden_state_index is None:
            return self.backbone(images)
        intermediates = self.backbone.forward_intermediates(
            images,
            intermediates_only=True,
        )  # type: ignore[attr-defined]
        hidden_state = intermediates[self._hidden_state_index]
        return _hidden_state_to_sequence(hidden_state)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Preprocessed image batch of shape ``(B, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(B, N_tokens, output_dim)``.
        """
        features = self._backbone_forward(images)

        if self._out_format == "pooled":
            # (B, D) -> (B, 1, D)
            features = features.unsqueeze(1)
        elif self._out_format == "spatial":
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if not self.use_all_patches:
                features = features[:, :1, :]
        else:
            # seq: (B, N, D)
            if not self.use_all_patches:
                features = features[:, :1, :]

        projected = self.proj(features)  # (B, N, output_dim)
        return self.norm(projected)
