from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_VOICE_DESIGN_CONFIG


class IrodoriVoiceDesignConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "VoiceDesignConfig"),
            display_name="IrodoriTTS VoiceDesign Config",
            category=CATEGORY,
            inputs=[
                io.String.Input(
                    "caption",
                    multiline=True,
                    tooltip="VoiceDesignモデル用の声質・話し方・感情などの説明文です。",
                ),
                io.Float.Input(
                    "cfg_scale_caption",
                    default=3.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="caption/style条件のCFG強度です。VoiceDesignモデルで有効です。",
                ),
                io.Int.Input(
                    "max_caption_len",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="caption token長の上限です。0ならチェックポイント既定値を使います。",
                ),
            ],
            outputs=[
                IO_VOICE_DESIGN_CONFIG.Output(display_name="irodori_voice_design_config"),
            ],
        )

    @classmethod
    def execute(cls, caption: str, cfg_scale_caption: float, max_caption_len: int):
        config = {
            "caption": str(caption).strip() or None,
            "cfg_scale_caption": float(cfg_scale_caption),
            "max_caption_len": None if max_caption_len <= 0 else int(max_caption_len),
        }
        return io.NodeOutput(config)
