from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .codec import patchify_latent
from .tokenizer import PretrainedTextTokenizer


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """
    Normalize latent tensor to (T, D).
    Accepts common layouts: (T, D), (D, T), (1, T, D), (1, D, T).
    """
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")

    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


class LatentTextDataset(Dataset):
    """
    Manifest format (JSONL), one sample per line:
      {"text": "...", "latent_path": "path/to/latent.pt", "speaker_id": "..."}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        latent_dim: int,
        max_latent_steps: int | None = None,
        subset_indices: list[int] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.latent_dim = latent_dim
        self.max_latent_steps = max_latent_steps
        subset_index_set: set[int] | None = None
        if subset_indices is not None:
            subset_index_set = {int(x) for x in subset_indices}
            if not subset_index_set:
                raise ValueError("subset_indices must contain at least one index.")

        self.samples: list[dict[str, Any]] = []
        self.speaker_to_indices: dict[str, list[int]] = {}
        for original_index, line in enumerate(
            self.manifest_path.read_text(encoding="utf-8").splitlines()
        ):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "text" not in item or "latent_path" not in item:
                raise ValueError(f"Invalid manifest line (needs text and latent_path): {line}")
            if subset_index_set is not None and original_index not in subset_index_set:
                continue
            self.samples.append(item)
            speaker_id = item.get("speaker_id")
            if speaker_id is not None:
                speaker_key = str(speaker_id)
                self.speaker_to_indices.setdefault(speaker_key, []).append(len(self.samples) - 1)

        if not self.samples:
            raise ValueError(f"No valid samples in manifest: {self.manifest_path}")

    def _resolve_latent_path(self, latent_path_raw: str) -> Path:
        latent_path = Path(latent_path_raw).expanduser()
        if not latent_path.is_absolute():
            latent_path = (self.manifest_dir / latent_path).resolve()
        return latent_path

    def _load_latent(self, latent_path_raw: str) -> torch.Tensor:
        latent_path = self._resolve_latent_path(latent_path_raw)
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = _coerce_latent_shape(latent, self.latent_dim).float()
        if self.max_latent_steps is not None:
            latent = latent[: self.max_latent_steps]
        return latent

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        latent = self._load_latent(item["latent_path"])

        ref_index = index
        speaker_id = item.get("speaker_id")
        has_speaker = speaker_id is not None
        if speaker_id is not None:
            speaker_key = str(speaker_id)
            candidates = self.speaker_to_indices.get(speaker_key, [])
            if len(candidates) > 1:
                alternatives = [i for i in candidates if i != index]
                if alternatives:
                    ref_index = random.choice(alternatives)

        if ref_index == index:
            ref_latent = latent
        else:
            ref_item = self.samples[ref_index]
            ref_latent = self._load_latent(ref_item["latent_path"])
        return {
            "text": item["text"],
            "latent": latent,
            "ref_latent": ref_latent,
            "has_speaker": has_speaker,
        }


@dataclass
class TTSCollator:
    tokenizer: PretrainedTextTokenizer
    latent_dim: int
    latent_patch_size: int
    fixed_target_latent_steps: int | None = None
    fixed_target_full_mask: bool = False
    max_text_len: int = 256

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        latents = [x["latent"] for x in batch]  # each: (T, D)
        ref_latents = [x["ref_latent"] for x in batch]  # each: (T_ref, D)
        has_speaker = torch.tensor([bool(x["has_speaker"]) for x in batch], dtype=torch.bool)
        bsz = len(latents)

        text_ids, text_mask = self.tokenizer.batch_encode(texts, max_length=self.max_text_len)

        if self.fixed_target_latent_steps is not None:
            max_t = int(self.fixed_target_latent_steps)
            if max_t <= 0:
                raise ValueError(
                    f"fixed_target_latent_steps must be > 0, got {self.fixed_target_latent_steps}"
                )
        else:
            max_t = max(x.shape[0] for x in latents)
        latent_batch = torch.zeros((bsz, max_t, self.latent_dim), dtype=torch.float32)
        latent_mask_valid = torch.zeros((bsz, max_t), dtype=torch.bool)
        for i, latent in enumerate(latents):
            n = min(latent.shape[0], max_t)
            latent_batch[i, :n] = latent[:n]
            latent_mask_valid[i, :n] = True
        latent_mask = latent_mask_valid.clone()
        if self.fixed_target_full_mask:
            latent_mask.fill_(True)

        max_ref_t = max(x.shape[0] for x in ref_latents)
        ref_batch = torch.zeros((bsz, max_ref_t, self.latent_dim), dtype=torch.float32)
        ref_mask = torch.zeros((bsz, max_ref_t), dtype=torch.bool)
        for i, ref_latent in enumerate(ref_latents):
            n = ref_latent.shape[0]
            ref_batch[i, :n] = ref_latent
            ref_mask[i, :n] = True

        latent_patched = patchify_latent(latent_batch, self.latent_patch_size)
        # Keep reference in latent-patched space. The model applies an extra
        # speaker_patch_size patching internally for speaker conditioning.
        ref_patched = patchify_latent(ref_batch, self.latent_patch_size)

        def _patch_mask(mask: torch.Tensor) -> torch.Tensor:
            if self.latent_patch_size <= 1:
                return mask
            usable = (mask.shape[1] // self.latent_patch_size) * self.latent_patch_size
            return (
                mask[:, :usable]
                .reshape(bsz, usable // self.latent_patch_size, self.latent_patch_size)
                .all(dim=-1)
            )

        latent_mask_patched = _patch_mask(latent_mask)
        latent_mask_valid_patched = _patch_mask(latent_mask_valid)
        ref_mask_patched = _patch_mask(ref_mask)

        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "latent": latent_batch,
            "latent_mask": latent_mask,
            "latent_mask_valid": latent_mask_valid,
            "latent_patched": latent_patched,
            "latent_mask_patched": latent_mask_patched,
            "latent_mask_valid_patched": latent_mask_valid_patched,
            "latent_padding_mask_patched": ~latent_mask_valid_patched,
            "ref_latent": ref_batch,
            "ref_latent_mask": ref_mask,
            "ref_latent_patched": ref_patched,
            "ref_latent_mask_patched": ref_mask_patched,
            "has_speaker": has_speaker,
        }
