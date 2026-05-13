from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_TRIM_TAIL_CONFIG


class IrodoriTrimTailConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "TrimTailConfig"),
            display_name="IrodoriTTS Trim Tail Config",
            category=CATEGORY,
            inputs=[
                io.Int.Input(
                    "tail_window_size",
                    default=20,
                    min=1,
                    max=200,
                    tooltip="末尾切り詰め判定に使う潜在窓サイズです。",
                ),
                io.Float.Input(
                    "tail_std_threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="末尾切り詰め判定の標準偏差しきい値です。",
                ),
                io.Float.Input(
                    "tail_mean_threshold",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="末尾切り詰め判定の平均値しきい値です。",
                ),
            ],
            outputs=[
                IO_TRIM_TAIL_CONFIG.Output(display_name="irodori_trim_tail_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        tail_window_size: int,
        tail_std_threshold: float,
        tail_mean_threshold: float,
    ):
        config = {
            "tail_window_size": int(tail_window_size),
            "tail_std_threshold": float(tail_std_threshold),
            "tail_mean_threshold": float(tail_mean_threshold),
        }
        return io.NodeOutput(config)
