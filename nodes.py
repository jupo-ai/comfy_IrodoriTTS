from comfy_api.latest import io
import os
import sys
import hashlib
import folder_paths

from .irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions
)


def _available_devices():
    return list_available_runtime_devices()

def _available_precisions(device="cuda"):
    try:
        return list_available_runtime_precisions(device)
    except:
        return ["fp32", "bf16"]


IO_IRODORI_MODEL = io.Custom("IRODORI_MODEL")
IO_IRODORI_REF_CONFIG = io.Custom("IRODORI_REF_CONFIG")
IO_IRODORI_CFG_CONFIG = io.Custom("IRODORI_CFG_CONFIG")
IO_IRODORI_RESCALE_CONFIG = io.Custom("IRODORI_RESCALE_CONFIG")

CATEGORY = "Irodori-TTS"



# ===============================================
# Irodori Model Loader
# ===============================================
class IrodoriTTSModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        model_files = folder_paths.get_filename_list("checkpoints")
        devices = _available_devices()
        precisions = _available_precisions()

        return io.Schema(
            node_id="IrodoriTTSModelLoader", 
            display_name="IrodoriTTS Model Loader", 
            category=CATEGORY, 
            inputs=[
                io.Combo.Input("model_name", options=model_files), 
                io.Combo.Input("model_device", options=devices), 
                io.Combo.Input("model_precision", options=precisions), 
                io.Combo.Input("codec_device", options=devices), 
                io.Combo.Input("codec_precision", options=precisions), 
                io.Boolean.Input("enable_watermark", default=False), 
            ], 
            outputs=[
                IO_IRODORI_MODEL.Output(display_name="irodori_model")
            ], 
        )
    
    @classmethod
    def execute(
        cls, 
        model_name: str, 
        model_device: str, 
        model_precision: str, 
        codec_device: str, 
        codec_precision: str, 
        enable_watermark: bool
    ):
        checkpoint_path = folder_paths.get_full_path("checkpoints", model_name)
        if not checkpoint_path:
            checkpoint_path = model_name
        
        runtime_key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=model_device,
            codec_repo="facebook/dacvae-watermarked",
            model_precision=model_precision,
            codec_device=codec_device,
            codec_precision=codec_precision,
            enable_watermark=enable_watermark,
            compile_model=False,
            compile_dynamic=False,
        )
        
        runtime, _ = get_cached_runtime(runtime_key)
        return io.NodeOutput(runtime)


class IrodoriTTSModelLoaderHF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        devices = _available_devices()
        precisions = _available_precisions()

        return io.Schema(
            node_id="IrodoriTTSModelLoaderHF", 
            display_name="IrodoriTTS Model Loader HF", 
            category=CATEGORY, 
            inputs=[
                io.String.Input("hf_checkpoint", default="Aratako/Irodori-TTS-500M"), 
                io.Combo.Input("model_device", options=devices), 
                io.Combo.Input("model_precision", options=precisions), 
                io.Combo.Input("codec_device", options=devices), 
                io.Combo.Input("codec_precision", options=precisions), 
                io.Boolean.Input("enable_watermark", default=False), 
            ], 
            outputs=[
                IO_IRODORI_MODEL.Output(display_name="irodori_model")
            ], 
        )
    
    @classmethod
    def execute(
        cls, 
        hf_checkpoint: str, 
        model_device: str, 
        model_precision: str, 
        codec_device: str, 
        codec_precision: str, 
        enable_watermark: bool
    ):
        from huggingface_hub import hf_hub_download
        
        repo_id = hf_checkpoint.strip()
        if not repo_id:
            raise ValueError("hf_checkpoint is required.")
        
        if repo_id.endswith(".pt") or repo_id.endswith(".safetensors"):
            checkpoint_path = repo_id
        else:
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        
        runtime_key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=model_device,
            codec_repo="facebook/dacvae-watermarked",
            model_precision=model_precision,
            codec_device=codec_device,
            codec_precision=codec_precision,
            enable_watermark=enable_watermark,
            compile_model=False,
            compile_dynamic=False,
        )
        
        runtime, _ = get_cached_runtime(runtime_key)
        return io.NodeOutput(runtime)


# ===============================================
# Irodori Reference Audio
# ===============================================
class IrodoriTTSReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])

        return io.Schema(
            node_id="IrodoriTTSReferenceAudio",
            display_name="IrodoriTTS Reference Audio",
            category=CATEGORY, 
            inputs=[
                io.Combo.Input("ref_audio", options=files), 
                io.Boolean.Input("normalize_ref_audio", default=False), 
                io.Float.Input("max_ref_seconds", default=30.0, min=1.0, max=120.0, step=1.0), 
                
            ], 
            outputs=[IO_IRODORI_REF_CONFIG.Output(display_name="ref_audio_config")],
        )

    @classmethod
    def execute(cls, ref_audio, normalize_ref_audio, max_ref_seconds):
        audio_path = folder_paths.get_annotated_filepath(ref_audio)
        config = {
            "ref_wav": audio_path, 
            "ref_normalize_db": -16.0 if normalize_ref_audio else None, 
            "ref_ensure_max": normalize_ref_audio, 
            "max_ref_seconds": max_ref_seconds, 
        }
        
        return io.NodeOutput(config)
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        ref_audio = kwargs.get("ref_audio")
        audio_path = folder_paths.get_annotated_filepath(ref_audio)
        m = hashlib.sha256()
        with open(audio_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()
    
    @classmethod
    def validate_inputs(cls, **kwargs):
        ref_audio = kwargs.get("ref_audio")
        if not folder_paths.exists_annotated_filepath(ref_audio):
            return "Invalid audio file: {}".format(ref_audio)
        return True
    

# ===============================================
# Irodori Advanced CFG
# ===============================================
class IrodoriTTSAdvancedCFG(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTSAdvancedCFG", 
            display_name="IrodoriTTS Advanced CFG", 
            category=CATEGORY, 
            inputs=[
                io.Float.Input("cfg_scale_override", default=-1.0, min=-1.0, max=10.0, step=0.1, tooltip="Set to > 0 to override"), 
                io.Float.Input("cfg_min_t", default=0.5, min=0.0, max=1.0, step=0.05), 
                io.Float.Input("cfg_max_t", default=1.0, min=0.0, max=1.0, step=0.05), 
            ], 
            outputs=[
                IO_IRODORI_CFG_CONFIG.Output(display_name="cfg_config"), 
            ], 
        )
    
    @classmethod
    def execute(cls, cfg_scale_override: float, cfg_min_t: float, cfg_max_t: float):
        config = {
            "cfg_scale_override": cfg_scale_override if cfg_scale_override > 0 else None, 
            "cfg_min_t": cfg_min_t, 
            "cfg_max_t": cfg_max_t, 
        }
        return io.NodeOutput(config)


# ===============================================
# Irodori Rescale Config
# ===============================================
class IrodoriTTSRescaleConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTSRescaleConfig", 
            display_name="IrodoriTTS Rescale Config", 
            category=CATEGORY, 
            inputs=[
                io.Float.Input(
                    "truncation_factor", 
                    default=-1.0, 
                    min=-1.0, 
                    max=1.0, 
                    step=0.05, 
                    tooltip="Set > 0 to enable"
                ), 
                io.Float.Input(
                    "rescale_k", 
                    default=-1.0, 
                    min=-1.0, 
                    max=10.0, 
                    step=0.1, 
                    tooltip="Set > 0 to enable. Typical range: 2.0 - 6.0"
                ), 
                io.Float.Input(
                    "rescale_sigma", 
                    default=-1.0, 
                    min=-1.0, 
                    max=3.0, 
                    step=0.01, 
                    tooltip="Set > 0 to enable (requires rescale_k). Typical range: 0.1 - 1.0"
                ), 
                io.Float.Input(
                    "speaker_kv_scale", 
                    default=-1.0, 
                    min=-1.0, 
                    max=5.0, 
                    step=0.05, 
                    tooltip="Set > 0 to scale speaker K/V strength. Typical range: 1.0 - 3.0"
                ), 
                io.Float.Input(
                    "speaker_kv_min_t", 
                    default=0.9, 
                    min=0.0, 
                    max=1.0, 
                    step=0.05, 
                    tooltip="KV scale is applied while t >= this value, then reverted"
                ), 
                io.Int.Input(
                    "speaker_kv_max_layers", 
                    default=-1, 
                    min=-1, 
                    max=32, 
                    step=1, 
                    tooltip="Max transformer layers to apply speaker_kv_scale to. -1 = all layers"
                ), 
            ], 
            outputs=[
                IO_IRODORI_RESCALE_CONFIG.Output(display_name="rescale_config"), 
            ], 
        )
    
    @classmethod
    def execute(cls, truncation_factor, rescale_k, rescale_sigma, speaker_kv_scale, speaker_kv_min_t, speaker_kv_max_layers):
        config = {
            "truncation_factor": truncation_factor if truncation_factor > 0 else None,
            "rescale_k": rescale_k if rescale_k > 0 else None,
            "rescale_sigma": rescale_sigma if rescale_sigma > 0 else None,
            "speaker_kv_scale": speaker_kv_scale if speaker_kv_scale > 0 else None,
            "speaker_kv_min_t": speaker_kv_min_t,
            "speaker_kv_max_layers": speaker_kv_max_layers if speaker_kv_max_layers >= 0 else None,
        }
        return io.NodeOutput(config)
    
    
# ===============================================
# Irodori Sampler
# ===============================================
class IrodoriTTSSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTSSampler", 
            display_name="IrodoriTTS Sampler", 
            category=CATEGORY, 
            inputs=[
                IO_IRODORI_MODEL.Input("model", display_name="irodori_model"), 
                io.String.Input("text", multiline=True), 
                io.Int.Input("seed", default=0, min=0, max=sys.maxsize), 
                io.Int.Input("num_steps", default=40, min=1, max=120), 
                io.Combo.Input("cfg_guidance_mode", options=["independent", "joint", "alternating"], default="independent"), 
                io.Float.Input("cfg_scale_text", default=3.0, min=0.0, max=10.0, step=0.1), 
                io.Float.Input("cfg_scale_speaker", default=5.0, min=0.0, max=10.0, step=0.1), 
                io.Boolean.Input("context_kv_cache", default=True), 
                
                IO_IRODORI_REF_CONFIG.Input("ref_audio_config", display_name="ref_audio_config", optional=True), 
                IO_IRODORI_CFG_CONFIG.Input("cfg_config", display_name="cfg_config", optional=True), 
                IO_IRODORI_RESCALE_CONFIG.Input("rescale_config", display_name="rescale_config", optional=True), 
            ], 
            outputs=[
                io.Audio.Output(), 
            ], 
        )
    
    @classmethod
    def execute(cls, model, text, seed, num_steps, cfg_guidance_mode, cfg_scale_text, cfg_scale_speaker, context_kv_cache, ref_audio_config={}, cfg_config={}, rescale_config={}):
        # Unpack optional configs
        ref_wav = ref_audio_config.get("ref_wav", None)
        no_ref = ref_wav == None
        ref_normalize_db = ref_audio_config.get("ref_normalize_db", None)
        ref_ensure_max = ref_audio_config.get("ref_ensure_max", False)
        max_ref_seconds = ref_audio_config.get("max_ref_seconds", 30.0)
        
        cfg_scale_override = cfg_config.get("cfg_scale_override", None)
        cfg_min_t = cfg_config.get("cfg_min_t", 0.5)
        cfg_max_t = cfg_config.get("cfg_max_t", 1.0)

        truncation_factor = rescale_config.get("truncation_factor", None)
        rescale_k = rescale_config.get("rescale_k", None)
        rescale_sigma = rescale_config.get("rescale_sigma", None)
        speaker_kv_scale = rescale_config.get("speaker_kv_scale", None)
        speaker_kv_min_t = rescale_config.get("speaker_kv_min_t", 0.9)
        speaker_kv_max_layers = rescale_config.get("speaker_kv_max_layers", None)

        req = SamplingRequest(
            text=text,
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=no_ref,
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=ref_ensure_max,
            num_candidates=1,
            decode_mode="sequential",
            seconds=30.0,
            max_ref_seconds=max_ref_seconds,
            max_text_len=None,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=cfg_guidance_mode,
            cfg_scale=cfg_scale_override,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=context_kv_cache,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            seed=seed,
            trim_tail=True,
        )
        
        result = model.synthesize(req, log_fn=print)
        
        audio_tensor = result.audio
        # Result is [channels, samples]. ComfyUI expects [batch, channels, samples]
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        out_audio = {"waveform": audio_tensor, "sample_rate": result.sample_rate}
        return io.NodeOutput(out_audio)


# ===============================================
# Irodori Emoji Selector
# ===============================================
class IrodoriTTSEmojiSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IrodoriTTSEmojiSelector", 
            display_name="IrodoriTTS Emoji Selector", 
            category=CATEGORY, 
            inputs=[], 
            outputs=[], 
        )
    
    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput()
    

