from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_CFG_CONFIG, none_if_non_positive


class IrodoriCFGConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "CFGConfig"),
            display_name="IrodoriTTS CFG Config",
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "cfg_guidance_mode",
                    options=["independent", "joint", "alternating"],
                    default="independent",
                    tooltip="CFGの条件合成方式です。通常はindependentを使用します。",
                ),
                io.Float.Input(
                    "cfg_scale_text",
                    default=3.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="読み上げテキスト条件のCFG強度です。大きいほどテキスト追従が強くなります。",
                ),
                io.Float.Input(
                    "cfg_scale_speaker",
                    default=5.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="参照話者条件のCFG強度です。大きいほど参照音声の話者性が強くなります。",
                ),
                io.Float.Input(
                    "cfg_scale_override",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="全CFG条件に共通の強度を指定します。0なら個別scaleを使用します。",
                ),
                io.Float.Input(
                    "cfg_min_t",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="CFGを適用する拡散時刻の下限です。通常は0.5で構いません。",
                ),
                io.Float.Input(
                    "cfg_max_t",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="CFGを適用する拡散時刻の上限です。通常は1.0で構いません。",
                ),
            ],
            outputs=[
                IO_CFG_CONFIG.Output(display_name="irodori_cfg_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        cfg_guidance_mode: str,
        cfg_scale_text: float,
        cfg_scale_speaker: float,
        cfg_scale_override: float,
        cfg_min_t: float,
        cfg_max_t: float,
    ):
        config = {
            "cfg_guidance_mode": str(cfg_guidance_mode),
            "cfg_scale_text": float(cfg_scale_text),
            "cfg_scale_speaker": float(cfg_scale_speaker),
            "cfg_scale_override": none_if_non_positive(float(cfg_scale_override)),
            "cfg_min_t": float(cfg_min_t),
            "cfg_max_t": float(cfg_max_t),
        }
        return io.NodeOutput(config)
