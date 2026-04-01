"""Helpers for building a normalized voice state contract."""
from __future__ import annotations


def _normalize_mode(mode: object) -> str:
    value = str(mode or "idle").strip().lower()
    return value or "idle"


def _normalize_context(context: object) -> dict:
    if not isinstance(context, dict):
        return {}
    return dict(context)


def _collect_known_labels(context: dict) -> list[str]:
    labels: list[str] = []

    def _add(label: object) -> None:
        if not isinstance(label, str):
            return
        cleaned = label.strip().lower()
        if cleaned and cleaned not in labels:
            labels.append(cleaned)

    _add(context.get("label"))
    _add(context.get("current_label"))

    known = context.get("known_labels")
    if isinstance(known, list):
        for label in known:
            _add(label)

    matches = context.get("scan_matches")
    if isinstance(matches, list):
        for match in matches:
            if isinstance(match, dict):
                _add(match.get("label"))

    return labels


def build_state_contract(mode: object, context: object) -> dict:
    """Return a stable, JSON-safe state payload for downstream voice helpers."""
    mode_value = _normalize_mode(mode)
    ctx = _normalize_context(context)

    contract_context = {
        "scan_id": str(ctx.get("scan_id", "") or ""),
        "label": str(ctx.get("current_label") or ctx.get("label") or ""),
        "item_index": ctx.get("item_index"),
        "onboarding_phase": str(ctx.get("onboarding_phase", "") or ""),
    }

    if "pending_action" in ctx:
        contract_context["pending_action"] = str(ctx.get("pending_action") or "")

    return {
        "current_mode": mode_value,
        "mode": mode_value,
        "context": contract_context,
        "known_labels": _collect_known_labels(ctx),
    }
