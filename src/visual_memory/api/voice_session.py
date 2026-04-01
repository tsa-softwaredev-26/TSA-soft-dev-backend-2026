"""Per-connection session state for the WebSocket voice interface."""
from __future__ import annotations

from dataclasses import dataclass, field

# States
# idle - default, waiting for any command
# awaiting_image - server needs a photo before it can run the pending action
# awaiting_location - after a successful teach, expecting a room name
# awaiting_confirmation - expecting yes/no on a pending action
# focused_on_item - last scan had matches; item context active for gestures/queries
# onboarding_teach - first-time flow: teaching items (1 or 2)
# onboarding_await_scan - first-time flow: both items taught, ready to scan
VALID_STATES = frozenset({
    "idle",
    "awaiting_image",
    "awaiting_location",
    "awaiting_confirmation",
    "focused_on_item",
    "onboarding_teach",
    "onboarding_await_scan",
})


@dataclass
class VoiceSession:
    sid: str
    state: str = "idle"
    # context keys used across handlers:
    #   pending_action str  - "scan" or "remember" (set when awaiting_image)
    #   label str - current item label
    #   scan_id str - last scan_id from ScanPipeline
    #   scan_matches list - cached match list from last scan
    #   item_index int  - current position in scan_matches
    #   current_label  str  - label at item_index
    #   onboarding_phase str  - "teach_1" | "teach_2" | "await_scan" | "ask" | "ask_prompted"
    #   onboarding_item_1 str  - label of first item taught during onboarding
    #   onboarding_item_2 str  - label of second item taught during onboarding
    #   onboarding_ask_label str - label the ask-demo should ask about
    #   scan_count int  - total scans this session (hint triggers)
    #   teach_count int  - total successful teaches this session
    #   hints_given set  - hint types already emitted this session
    context: dict = field(default_factory=dict)


_sessions: dict[str, VoiceSession] = {}

def get_session(sid: str) -> VoiceSession:
    if sid not in _sessions:
        _sessions[sid] = VoiceSession(sid=sid)
    return _sessions[sid]


def clear_session(sid: str) -> None:
    _sessions.pop(sid, None)
