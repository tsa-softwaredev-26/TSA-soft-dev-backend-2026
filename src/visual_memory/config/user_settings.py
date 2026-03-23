from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visual_memory.database.store import DatabaseStore


# -----------------------------------------------------------------------

class PerformanceMode(str, Enum):
    FAST     = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass
class PerformanceConfig:
    """Runtime parameters implied by each PerformanceMode.

    Not user-facing - the mobile app only exposes the mode name.
    Values here document what each mode means to the pipeline.
    """
    depth_enabled: bool
    target_latency: float  # seconds (advisory; pipeline does not enforce)

    @staticmethod
    def for_mode(mode: PerformanceMode) -> PerformanceConfig:
        _MAP = {
            PerformanceMode.FAST: PerformanceConfig(
                depth_enabled=False,
                target_latency=1.0,
            ),
            PerformanceMode.BALANCED: PerformanceConfig(
                depth_enabled=True,
                target_latency=2.0,
            ),
            PerformanceMode.ACCURATE: PerformanceConfig(
                depth_enabled=True,
                target_latency=4.0,
            ),
        }
        return _MAP[mode]


# -----------------------------------------------------------------------

# Valid values for button_layout (stub - mobile UI not yet shipped)
_BUTTON_LAYOUTS = {"default", "swapped"}


@dataclass
class UserSettings:
    """User-facing preferences persisted across sessions.

    Separate from config/settings.py which holds ML tuning params.
    Persisted as JSON at data/user_settings.json by default.
    """

    # -- visible settings (shown in app Settings screen) ----------------
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    voice_speed: float = 1.0          # 0.5 (slow) - 2.0 (fast)

    # -- advanced settings (collapsed by default in UI) -----------------
    auto_update_location: bool = False  # prompt to update location after found
    learning_enabled: bool = True       # collect feedback + improve with use

    # -- accessibility stub (not yet exposed in UI) ---------------------
    # Controls physical button positions on the home screen.
    # "default" = scan left / remember right.
    # "swapped" = remember left / scan right.
    # Extend this field when the mobile layout ships.
    button_layout: str = "default"

    # -------------------------------------------------------------------

    def get_performance_config(self) -> PerformanceConfig:
        return PerformanceConfig.for_mode(self.performance_mode)

    def save(self, db: DatabaseStore) -> None:
        db.save_user_settings({
            "performance_mode": self.performance_mode.value,
            "voice_speed": self.voice_speed,
            "auto_update_location": self.auto_update_location,
            "learning_enabled": self.learning_enabled,
            "button_layout": self.button_layout,
        })

    @classmethod
    def load(cls, db: DatabaseStore) -> UserSettings:
        data = db.load_user_settings() or {}
        try:
            return cls(
                performance_mode=PerformanceMode(data.get("performance_mode", "balanced")),
                voice_speed=float(data.get("voice_speed", 1.0)),
                auto_update_location=bool(data.get("auto_update_location", False)),
                learning_enabled=bool(data.get("learning_enabled", True)),
                button_layout=str(data.get("button_layout", "default")),
            )
        except ValueError:
            return cls()

    def to_dict(self) -> dict:
        cfg = self.get_performance_config()
        return {
            "performance_mode": self.performance_mode.value,
            "voice_speed": self.voice_speed,
            "auto_update_location": self.auto_update_location,
            "learning_enabled": self.learning_enabled,
            "button_layout": self.button_layout,
            # read-only derived fields (informational for client)
            "performance_config": {
                "depth_enabled": cfg.depth_enabled,
                "target_latency": cfg.target_latency,
            },
        }
