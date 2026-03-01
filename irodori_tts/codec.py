from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert latent from (B, T, D) -> (B, T//patch, D*patch).
    Extra tail tokens are dropped.
    """
    if patch_size <= 1:
        return latent
    bsz, seq_len, dim = latent.shape
    usable = (seq_len // patch_size) * patch_size
    latent = latent[:, :usable]
    latent = latent.reshape(bsz, usable // patch_size, dim * patch_size)
    return latent


def unpatchify_latent(patched: torch.Tensor, patch_size: int, latent_dim: int) -> torch.Tensor:
    """
    Convert latent from (B, T_p, D*patch) -> (B, T_p*patch, D).
    """
    if patch_size <= 1:
        return patched
    return patched.reshape(patched.shape[0], patched.shape[1] * patch_size, latent_dim)


@dataclass
class DACVAECodec:
    model: torch.nn.Module
    sample_rate: int
    latent_dim: int
    device: torch.device
    dtype: torch.dtype
    enable_watermark: bool
    watermark_alpha: float | None

    @classmethod
    def load(
        cls,
        repo_id: str = "facebook/dacvae-watermarked",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        enable_watermark: bool = False,
        watermark_alpha: float | None = None,
    ) -> DACVAECodec:
        # Prefer installed package; fallback to local clone at ../dacvae.
        try:
            from dacvae import DACVAE
        except ImportError:
            local_repo = Path(__file__).resolve().parents[2] / "dacvae"
            if local_repo.exists():
                sys.path.insert(0, str(local_repo))
            from dacvae import DACVAE

        model = DACVAE.load(repo_id).eval().to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)

        configured_watermark_alpha: float | None = None
        configured_enable_watermark = False
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "alpha"):
            default_alpha = float(decoder.alpha)
            if watermark_alpha is not None:
                target_alpha = float(watermark_alpha)
            elif enable_watermark:
                target_alpha = default_alpha
            else:
                target_alpha = 0.0
            decoder.alpha = float(target_alpha)
            configured_watermark_alpha = float(decoder.alpha)
            configured_enable_watermark = configured_watermark_alpha > 0.0
            if not configured_enable_watermark and hasattr(decoder, "wm_model"):
                # Keep decode output mono while skipping heavy watermark encode/decode path.
                def _watermark_passthrough(
                    x: torch.Tensor,
                    message: torch.Tensor | None = None,
                    _decoder=decoder,
                ) -> torch.Tensor:
                    del message
                    return _decoder.wm_model.encoder_block.forward_no_conv(x)

                decoder.watermark = _watermark_passthrough

        model_dtype = next(model.parameters()).dtype
        # Infer latent dimension by encoding a tiny random signal.
        dummy = torch.zeros(1, 1, 2048, device=device, dtype=model_dtype)
        with torch.inference_mode():
            z = model.encode(dummy)  # (B, D, T)
        return cls(
            model=model,
            sample_rate=int(model.sample_rate),
            latent_dim=int(z.shape[1]),
            device=torch.device(device),
            dtype=model_dtype,
            enable_watermark=configured_enable_watermark,
            watermark_alpha=configured_watermark_alpha,
        )

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        normalize_db: float | None = None,
        ensure_max: bool = False,
    ) -> torch.Tensor:
        """
        Input:
          waveform: (B, C, T) or (C, T)
          normalize_db: Optional target loudness (LUFS-like dB) applied before encode
          ensure_max: If True, scale down only when abs peak exceeds 1.0
        Output:
          latent: (B, T_latent, D_latent)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform ndim=3, got shape={tuple(waveform.shape)}")

        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        waveform = waveform.to(dtype=torch.float32)
        if normalize_db is not None:
            try:
                loudness = torchaudio.functional.loudness(waveform, self.sample_rate)
                if loudness.ndim == 0:
                    loudness = loudness.unsqueeze(0)
                gain_db = (float(normalize_db) - loudness).clamp(min=-80.0, max=80.0)
                gain = torch.pow(
                    torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype),
                    gain_db / 20.0,
                ).view(-1, 1, 1)
                finite_mask = torch.isfinite(gain)
                waveform = torch.where(finite_mask, waveform * gain, waveform)
            except Exception:
                # Keep behavior robust when loudness calculation is unavailable for an input.
                pass

        if ensure_max:
            peak = waveform.abs().amax(dim=-1, keepdim=True).amax(dim=1, keepdim=True)
            safe_peak = peak.clamp(min=1.0)
            waveform = waveform / safe_peak

        waveform = waveform.to(self.device, dtype=self.dtype)
        encoded = self.model.encode(waveform)  # (B, D, T)
        return encoded.transpose(1, 2).contiguous()  # (B, T, D)

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Input:
          latent: (B, T, D)
        Output:
          audio: (B, 1, samples)
        """
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=self.dtype)  # (B, D, T)
        return self.model.decode(z)

    def encode_file(self, path: str | Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(str(path))
        except RuntimeError:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            wav = torch.from_numpy(data)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.T
        wav = wav.unsqueeze(0)  # (1, C, T)
        return self.encode_waveform(wav, sr).cpu()
