import sys

import comfy.utils
import torch
from comfy_api.latest import io

from ...modules.irodori_character_voice.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    get_cached_runtime,
    offload_cached_runtime,
)
from ...node_utils import mk_name
from ..wrapper.common import CATEGORY, PACKAGE_NAME
from ..wrapper.irodori_common import (
    IO_CFG_CONFIG,
    IO_MODEL_CONFIG,
    IO_RESCALE_CONFIG,
    IO_TRIM_TAIL_CONFIG,
    none_if_non_positive,
)


class IrodoriCharacterVoiceSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "CharacterVoiceSampler"),
            display_name="Irodori Character Voice Sampler",
            category=f"{CATEGORY}/Character Voice",
            inputs=[
                IO_MODEL_CONFIG.Input(
                    "model_config",
                    display_name="irodori_model_config",
                    tooltip="IrodoriTTS Model Loaderの出力を接続します。",
                ),
                io.Image.Input(
                    "character_image",
                    optional=True,
                    tooltip="声質や話し方の条件に使うキャラクター画像です。未接続の場合は画像条件なしで生成します。バッチ画像の場合は先頭の1枚を使用します。",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip="生成する読み上げテキストです。",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="生成に使用する乱数seedです。同じ設定なら同じ結果を再現します。",
                ),
                io.Float.Input(
                    "seconds",
                    default=30.0,
                    min=1.0,
                    max=120.0,
                    step=0.5,
                    tooltip="生成する音声長です。Character Voiceのデモ実装では30秒固定が標準です。",
                ),
                io.Int.Input(
                    "num_steps",
                    default=40,
                    min=1,
                    max=120,
                    tooltip="サンプリングステップ数です。大きいほど遅くなります。",
                ),
                IO_CFG_CONFIG.Input(
                    "cfg_config",
                    optional=True,
                    tooltip="CFGの詳細設定です。cfg_scale_speakerとcfg_scale_captionはCharacter Voiceでは使用しません。",
                ),
                IO_RESCALE_CONFIG.Input(
                    "rescale_config",
                    optional=True,
                    tooltip="rescale補正の設定です。speaker K/V補正はCharacter Voiceでは使用しません。",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=16,
                    tooltip="同じ条件で同時生成する音声のバッチ数です。出力AUDIOのbatch方向に複数候補を格納します。",
                ),
                io.Combo.Input(
                    "decode_mode",
                    options=["sequential", "batch"],
                    default="sequential",
                    tooltip="codecデコード方式です。batchは速い場合がありますがVRAMを多く使います。",
                ),
                io.Boolean.Input(
                    "context_kv_cache",
                    default=True,
                    tooltip="コンテキストK/Vキャッシュを使用します。通常は有効を推奨します。",
                ),
                io.Int.Input(
                    "max_text_len",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="テキストtoken長の上限です。0ならチェックポイント既定値を使います。",
                ),
                io.Boolean.Input(
                    "trim_tail",
                    default=True,
                    tooltip="末尾の無音や平坦化した部分を推定して切り詰めます。",
                ),
                IO_TRIM_TAIL_CONFIG.Input(
                    "trim_tail_config",
                    optional=True,
                    tooltip="末尾切り詰め判定の詳細設定です。未接続なら標準値を使用します。",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config: dict,
        text: str,
        seed: int,
        seconds: float,
        num_steps: int,
        batch_size: int,
        decode_mode: str,
        context_kv_cache: bool,
        max_text_len: int,
        trim_tail: bool,
        character_image: torch.Tensor | None = None,
        cfg_config: dict | None = None,
        rescale_config: dict | None = None,
        trim_tail_config: dict | None = None,
    ):
        cfg_config = cfg_config or {}
        rescale_config = rescale_config or {}
        trim_tail_config = trim_tail_config or {}

        runtime_key = RuntimeKey(
            checkpoint=str(model_config["checkpoint"]),
            model_device=str(model_config.get("model_device", "cuda")),
            codec_repo=str(model_config["codec_repo"]),
            model_precision=str(model_config.get("model_precision", "fp32")),
            codec_device=str(model_config.get("codec_device", "cpu")),
            codec_precision=str(model_config.get("codec_precision", "fp32")),
            enable_watermark=bool(model_config.get("enable_watermark", False)),
            compile_model=bool(model_config.get("compile_model", False)),
            compile_dynamic=bool(model_config.get("compile_dynamic", False)),
        )

        cfg_guidance_mode = cfg_config.get("cfg_guidance_mode", "independent")
        cfg_scale_override = cfg_config.get("cfg_scale_override", None)
        if cfg_scale_override is None and str(cfg_guidance_mode).strip().lower() == "joint":
            cfg_scale_override = float(cfg_config.get("cfg_scale_text", 3.0))
        req = SamplingRequest(
            text=str(text),
            caption=None,
            character_image=character_image,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=int(batch_size),
            decode_mode=str(decode_mode),
            seconds=float(seconds),
            max_ref_seconds=30.0,
            max_text_len=none_if_non_positive(int(max_text_len)),
            max_caption_len=None,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_config.get("cfg_scale_text", 3.0)),
            cfg_scale_caption=0.0,
            cfg_scale_speaker=0.0,
            cfg_scale_character=float(cfg_config.get("cfg_scale_character", 3.0)),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale=cfg_scale_override,
            cfg_min_t=float(cfg_config.get("cfg_min_t", 0.5)),
            cfg_max_t=float(cfg_config.get("cfg_max_t", 1.0)),
            truncation_factor=rescale_config.get("truncation_factor", None),
            rescale_k=rescale_config.get("rescale_k", None),
            rescale_sigma=rescale_config.get("rescale_sigma", None),
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            seed=int(seed),
            trim_tail=bool(trim_tail),
            tail_window_size=int(trim_tail_config.get("tail_window_size", 20)),
            tail_std_threshold=float(trim_tail_config.get("tail_std_threshold", 0.05)),
            tail_mean_threshold=float(trim_tail_config.get("tail_mean_threshold", 0.1)),
        )

        progress_bar = comfy.utils.ProgressBar(int(num_steps))

        def update_progress(current: int, total: int) -> None:
            progress_bar.update_absolute(int(current), int(total))

        cache_policy = str(model_config.get("runtime_cache_policy", "offload_after_use"))
        try:
            runtime, _ = get_cached_runtime(runtime_key)
            if not runtime.model_cfg.use_character_condition:
                raise ValueError(
                    "Loaded checkpoint does not enable character conditioning. "
                    "Use a Character Voice checkpoint."
                )
            result = runtime.synthesize(req, log_fn=print, progress_callback=update_progress)
        finally:
            if cache_policy == "unload_after_use":
                clear_cached_runtime()
            elif cache_policy == "offload_after_use":
                offload_cached_runtime()
            elif cache_policy != "keep_gpu":
                print(
                    f"[Irodori Character Voice] unknown runtime_cache_policy={cache_policy!r}; "
                    "keeping runtime on GPU."
                )

        audios = result.audios or [result.audio]
        max_samples = max(int(audio.shape[-1]) for audio in audios)
        padded_audios = []
        for audio in audios:
            if audio.dim() != 2:
                raise ValueError(f"Expected generated audio shape [channels, samples], got {tuple(audio.shape)}")
            if int(audio.shape[-1]) < max_samples:
                audio = torch.nn.functional.pad(audio, (0, max_samples - int(audio.shape[-1])))
            padded_audios.append(audio)

        audio_tensor = torch.stack(padded_audios, dim=0)
        out_audio = {"waveform": audio_tensor, "sample_rate": result.sample_rate}
        return io.NodeOutput(out_audio)
