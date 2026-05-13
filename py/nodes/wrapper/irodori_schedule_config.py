from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_SCHEDULE_CONFIG


class IrodoriScheduleConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "ScheduleConfig"),
            display_name="IrodoriTTS Schedule Config",
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "schedule_mode",
                    options=["linear", "sway"],
                    default="linear",
                    tooltip="RF Eulerサンプリングの時刻スケジュールです。通常はlinearを使用します。",
                ),
                io.Float.Input(
                    "sway_coeff",
                    default=-1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    tooltip="schedule_modeがswayのときに時刻スケジュールを曲げる係数です。",
                ),
            ],
            outputs=[
                IO_SCHEDULE_CONFIG.Output(display_name="irodori_schedule_config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        schedule_mode: str,
        sway_coeff: float,
    ):
        config = {
            "t_schedule_mode": str(schedule_mode).strip().lower(),
            "sway_coeff": float(sway_coeff),
        }
        return io.NodeOutput(config)
