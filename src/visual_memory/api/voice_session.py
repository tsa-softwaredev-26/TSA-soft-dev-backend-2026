"""Per-connection session state and transition helpers for voice WS."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock

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
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
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
_sessions_lock = Lock()

_SESSION_IDLE_EXPIRY_S = 60 * 45
_SCAN_CONTEXT_TTL_S = 60 * 15
_PENDING_ACTION_TTL_S = 60 * 5
_ONBOARDING_TTL_S = 60 * 30


def _now() -> float:
    return time.monotonic()


def _touch(session: VoiceSession) -> None:
    session.updated_at = _now()


def _is_state(value: str) -> bool:
    return value in VALID_STATES


def _prune_stale_context(session: VoiceSession) -> None:
    now = _now()
    scan_ts = float(session.context.get("_scan_ts") or 0.0)
    if scan_ts and (now - scan_ts) > _SCAN_CONTEXT_TTL_S:
        for key in ("scan_matches", "scan_id", "item_index", "current_label", "_scan_ts"):
            session.context.pop(key, None)

    pending_ts = float(session.context.get("_pending_ts") or 0.0)
    if pending_ts and (now - pending_ts) > _PENDING_ACTION_TTL_S:
        for key in ("pending_action", "describe_target", "label", "_pending_ts"):
            session.context.pop(key, None)
        if session.state == "awaiting_image":
            session.state = "idle"

    onboarding_ts = float(session.context.get("_onboarding_ts") or 0.0)
    if onboarding_ts and (now - onboarding_ts) > _ONBOARDING_TTL_S:
        for key in ("onboarding_phase", "onboarding_item_1", "onboarding_item_2", "onboarding_ask_label", "_onboarding_ts"):
            session.context.pop(key, None)
        if session.state in {"onboarding_teach", "onboarding_await_scan"}:
            session.state = "idle"


def session_state_payload(session: VoiceSession) -> dict:
    return {
        "current_mode": session.state,
        "context": {
            "scan_id": session.context.get("scan_id", ""),
            "label": session.context.get("current_label", ""),
            "item_index": session.context.get("item_index"),
            "onboarding_phase": session.context.get("onboarding_phase", ""),
        },
    }


def apply_state_transition(
    session: VoiceSession,
    *,
    next_state: str | None = None,
    context_updates: dict | None = None,
    clear_keys: tuple[str, ...] = (),
) -> None:
    if next_state is not None and _is_state(next_state):
        session.state = next_state
    for key in clear_keys:
        session.context.pop(key, None)
    if context_updates:
        for key, value in context_updates.items():
            if value is None:
                session.context.pop(key, None)
            else:
                session.context[key] = value
    if any(k in (context_updates or {}) for k in ("scan_matches", "scan_id", "item_index", "current_label")):
        session.context["_scan_ts"] = _now()
    if any(k in (context_updates or {}) for k in ("pending_action", "describe_target", "label")):
        session.context["_pending_ts"] = _now()
    if any(k in (context_updates or {}) for k in ("onboarding_phase", "onboarding_item_1", "onboarding_item_2", "onboarding_ask_label")):
        session.context["_onboarding_ts"] = _now()
    _touch(session)


def focused_label(session: VoiceSession) -> str:
    return str(session.context.get("current_label") or "")


def focused_scan_id(session: VoiceSession) -> str:
    return str(session.context.get("scan_id") or "")


def focused_scan_index(session: VoiceSession) -> int:
    try:
        return int(session.context.get("item_index") or 0)
    except (TypeError, ValueError):
        return 0


def set_focused_match(session: VoiceSession, matches: list[dict], index: int) -> dict:
    safe_index = max(0, min(index, len(matches) - 1))
    match = matches[safe_index]
    apply_state_transition(
        session,
        next_state="focused_on_item",
        context_updates={
            "item_index": safe_index,
            "current_label": match.get("label", ""),
            "scan_matches": matches,
        },
    )
    return match


def clear_focus(session: VoiceSession) -> None:
    apply_state_transition(
        session,
        clear_keys=("scan_matches", "scan_id", "item_index", "current_label", "_scan_ts"),
    )

def get_session(sid: str) -> VoiceSession:
    with _sessions_lock:
        now = _now()
        stale_sids = [key for key, sess in _sessions.items() if (now - sess.updated_at) > _SESSION_IDLE_EXPIRY_S]
        for stale_sid in stale_sids:
            _sessions.pop(stale_sid, None)
        if sid not in _sessions:
            _sessions[sid] = VoiceSession(sid=sid)
        session = _sessions[sid]
        _prune_stale_context(session)
        _touch(session)
        return session


def clear_session(sid: str) -> None:
    with _sessions_lock:
        _sessions.pop(sid, None)
