from __future__ import annotations

import torch

from .model import TextToLatentRFDiT


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # MPS generators are not available on some PyTorch builds; use CPU generator as fallback.
    try:
        return torch.Generator(device=device).manual_seed(seed), device
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed), torch.device("cpu")


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def rf_predict_x0(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x_t = x0 + t * v  =>  x0 = x_t - t * v
    return x_t - t[:, None, None] * v_pred


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        _, _, k_speaker, v_speaker = context_kv_cache[i]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


@torch.inference_mode()
def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor,
    ref_mask: torch.Tensor,
    sequence_length: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
) -> torch.Tensor:
    """
    Euler sampling over RF ODE with text+reference conditioning CFG.

    Returns:
      latent sequence in patched space, shape (B, sequence_length, patched_latent_dim)
    """
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    x_t = torch.randn(
        (batch_size, sequence_length, latent_dim), device=rng_device, dtype=dtype, generator=rng
    )
    if rng_device != device:
        x_t = x_t.to(device=device)
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device) * init_scale
    has_text_cfg = cfg_scale_text > 0
    has_speaker_cfg = cfg_scale_speaker > 0
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    text_state_cond, text_mask_cond, speaker_state_cond, speaker_mask_cond = (
        model.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
        )
    )
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = torch.zeros_like(speaker_state_cond)
    speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

    cfg_batch_mult = 1
    text_state_cfg = text_state_cond
    text_mask_cfg = text_mask_cond
    speaker_state_cfg = speaker_state_cond
    speaker_mask_cfg = speaker_mask_cond
    if use_independent_cfg:
        if has_text_cfg and has_speaker_cfg:
            cfg_batch_mult = 3
            text_state_cfg = torch.cat([text_state_cond, text_state_uncond, text_state_cond], dim=0)
            text_mask_cfg = torch.cat([text_mask_cond, text_mask_uncond, text_mask_cond], dim=0)
            speaker_state_cfg = torch.cat(
                [speaker_state_cond, speaker_state_cond, speaker_state_uncond], dim=0
            )
            speaker_mask_cfg = torch.cat(
                [speaker_mask_cond, speaker_mask_cond, speaker_mask_uncond], dim=0
            )
        elif has_text_cfg:
            cfg_batch_mult = 2
            text_state_cfg = torch.cat([text_state_cond, text_state_uncond], dim=0)
            text_mask_cfg = torch.cat([text_mask_cond, text_mask_uncond], dim=0)
            speaker_state_cfg = torch.cat([speaker_state_cond, speaker_state_cond], dim=0)
            speaker_mask_cfg = torch.cat([speaker_mask_cond, speaker_mask_cond], dim=0)
        elif has_speaker_cfg:
            cfg_batch_mult = 2
            text_state_cfg = torch.cat([text_state_cond, text_state_cond], dim=0)
            text_mask_cfg = torch.cat([text_mask_cond, text_mask_cond], dim=0)
            speaker_state_cfg = torch.cat([speaker_state_cond, speaker_state_uncond], dim=0)
            speaker_mask_cfg = torch.cat([speaker_mask_cond, speaker_mask_uncond], dim=0)

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_uncond_text = None
    context_kv_uncond_speaker = None
    context_kv_uncond_joint = None
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=text_state_cfg,
                speaker_state=speaker_state_cfg,
            )
        elif use_joint_cfg:
            if has_text_cfg or has_speaker_cfg:
                context_kv_uncond_joint = model.build_context_kv_cache(
                    text_state=text_state_uncond,
                    speaker_state=speaker_state_uncond,
                )
        elif use_alternating_cfg:
            if has_text_cfg:
                context_kv_uncond_text = model.build_context_kv_cache(
                    text_state=text_state_uncond,
                    speaker_state=speaker_state_cond,
                )
            if has_speaker_cfg:
                context_kv_uncond_speaker = model.build_context_kv_cache(
                    text_state=text_state_cond,
                    speaker_state=speaker_state_uncond,
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        if context_kv_uncond_text is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_uncond_text,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    for i in range(num_steps):
        t = t_schedule[i]
        t_next = t_schedule[i + 1]
        tt = torch.full((batch_size,), t, device=device, dtype=dtype)

        use_cfg = (cfg_scale_text > 0 or cfg_scale_speaker > 0) and (
            cfg_min_t <= t.item() <= cfg_max_t
        )
        if use_cfg:
            if use_independent_cfg:
                x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
                tt_cfg = tt.repeat(cfg_batch_mult)
                v_out = model.forward_with_encoded_conditions(
                    x_t=x_t_cfg,
                    t=tt_cfg,
                    text_state=text_state_cfg,
                    text_mask=text_mask_cfg,
                    speaker_state=speaker_state_cfg,
                    speaker_mask=speaker_mask_cfg,
                    context_kv_cache=context_kv_cfg,
                )

                if has_text_cfg and has_speaker_cfg:
                    v_cond, v_uncond_text, v_uncond_speaker = v_out.chunk(3, dim=0)
                    v = (
                        v_cond
                        + cfg_scale_text * (v_cond - v_uncond_text)
                        + cfg_scale_speaker * (v_cond - v_uncond_speaker)
                    )
                elif has_text_cfg:
                    v_cond, v_uncond_text = v_out.chunk(2, dim=0)
                    v = v_cond + cfg_scale_text * (v_cond - v_uncond_text)
                else:
                    v_cond, v_uncond_speaker = v_out.chunk(2, dim=0)
                    v = v_cond + cfg_scale_speaker * (v_cond - v_uncond_speaker)
            else:
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    context_kv_cache=context_kv_cond,
                )
                if use_joint_cfg:
                    if has_text_cfg and has_speaker_cfg:
                        if abs(float(cfg_scale_text) - float(cfg_scale_speaker)) > 1e-6:
                            raise ValueError(
                                "cfg_guidance_mode='joint' expects a single guidance scale; "
                                "set equal text/speaker scales or use --cfg-scale."
                            )
                        joint_scale = float(cfg_scale_text)
                    elif has_text_cfg:
                        joint_scale = float(cfg_scale_text)
                    else:
                        joint_scale = float(cfg_scale_speaker)
                    v_uncond_joint = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=text_state_uncond,
                        text_mask=text_mask_uncond,
                        speaker_state=speaker_state_uncond,
                        speaker_mask=speaker_mask_uncond,
                        context_kv_cache=context_kv_uncond_joint,
                    )
                    v = v_cond + joint_scale * (v_cond - v_uncond_joint)
                elif use_alternating_cfg:
                    if has_text_cfg and has_speaker_cfg:
                        use_text_uncond = (i % 2) == 0
                    else:
                        use_text_uncond = has_text_cfg
                    if use_text_uncond:
                        alt_scale = float(cfg_scale_text)
                        v_uncond_alt = model.forward_with_encoded_conditions(
                            x_t=x_t.to(dtype),
                            t=tt,
                            text_state=text_state_uncond,
                            text_mask=text_mask_uncond,
                            speaker_state=speaker_state_cond,
                            speaker_mask=speaker_mask_cond,
                            context_kv_cache=context_kv_uncond_text,
                        )
                    else:
                        alt_scale = float(cfg_scale_speaker)
                        v_uncond_alt = model.forward_with_encoded_conditions(
                            x_t=x_t.to(dtype),
                            t=tt,
                            text_state=text_state_cond,
                            text_mask=text_mask_cond,
                            speaker_state=speaker_state_uncond,
                            speaker_mask=speaker_mask_uncond,
                            context_kv_cache=context_kv_uncond_speaker,
                        )
                    v = v_cond + alt_scale * (v_cond - v_uncond_alt)
                else:
                    raise RuntimeError(f"Unexpected cfg_guidance_mode: {cfg_guidance_mode}")
        else:
            v = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask=speaker_mask_cond,
                context_kv_cache=context_kv_cond,
            )

        if rescale_k is not None and rescale_sigma is not None:
            v = temporal_score_rescale(
                v_pred=v,
                x_t=x_t,
                t=t,
                rescale_k=float(rescale_k),
                rescale_sigma=float(rescale_sigma),
            )

        if (
            speaker_kv_active
            and speaker_kv_min_t is not None
            and (t_next < speaker_kv_min_t)
            and (t >= speaker_kv_min_t)
        ):
            inv_scale = 1.0 / float(speaker_kv_scale)
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cond,
                scale=inv_scale,
                max_layers=speaker_kv_max_layers,
            )
            if context_kv_cfg is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cfg,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            if context_kv_uncond_text is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_uncond_text,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
            speaker_kv_active = False

        x_t = x_t + v * (t_next - t)

    return x_t
