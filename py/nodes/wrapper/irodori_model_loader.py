import folder_paths
from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import (
    IO_MODEL_CONFIG,
    available_devices,
    available_precisions,
    codec_repo_for_latent_dim,
    peek_latent_dim_from_checkpoint,
    resolve_checkpoint_path,
)


class IrodoriModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        model_files = folder_paths.get_filename_list("checkpoints")
        devices = available_devices()
        precisions = available_precisions()

        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "ModelLoader"),
            display_name="IrodoriTTS Model Loader",
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "model",
                    options=model_files,
                    tooltip="使用するIrodoriTTSのチェックポイントを選択します。",
                ),
                io.Combo.Input(
                    "model_device",
                    options=devices,
                    tooltip="TTSモデル本体を実行するデバイスです。通常はcudaを推奨します。",
                ),
                io.Combo.Input(
                    "model_precision",
                    options=precisions,
                    tooltip="TTSモデル本体の計算精度です。cudaではbf16でVRAMを節約できます。",
                ),
                io.Combo.Input(
                    "codec_device",
                    options=devices,
                    tooltip="DACVAE codecを実行するデバイスです。VRAM節約時はcpuも選べます。",
                ),
                io.Combo.Input(
                    "codec_precision",
                    options=precisions,
                    tooltip="DACVAE codecの計算精度です。cpuではfp32を使用してください。",
                ),
                io.Boolean.Input(
                    "enable_watermark",
                    default=False,
                    tooltip="codec側のウォーターマーク処理を有効にします。通常は無効で構いません。",
                ),
                io.Boolean.Input(
                    "compile_model",
                    default=False,
                    tooltip="torch.compileで推論関数をコンパイルします。初回が遅くなり、環境によっては不安定です。",
                ),
                io.Boolean.Input(
                    "compile_dynamic",
                    default=False,
                    tooltip="torch.compileのdynamicモードを有効にします。入力長が変わる場合向けですが、通常は無効で構いません。",
                ),
            ],
            outputs=[
                IO_MODEL_CONFIG.Output(display_name="irodori_model_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        model_device: str,
        model_precision: str,
        codec_device: str,
        codec_precision: str,
        enable_watermark: bool,
        compile_model: bool,
        compile_dynamic: bool,
    ):
        checkpoint_path = resolve_checkpoint_path(model)
        latent_dim = peek_latent_dim_from_checkpoint(checkpoint_path)
        config = {
            "checkpoint": checkpoint_path,
            "latent_dim": latent_dim,
            "model_device": model_device,
            "codec_repo": codec_repo_for_latent_dim(latent_dim),
            "model_precision": model_precision,
            "codec_device": codec_device,
            "codec_precision": codec_precision,
            "enable_watermark": bool(enable_watermark),
            "compile_model": bool(compile_model),
            "compile_dynamic": bool(compile_dynamic),
        }
        return io.NodeOutput(config)
