from __future__ import annotations

from typing import Any

from flask import Request
from werkzeug.exceptions import BadRequest


def read_json_dict(request: Request, *, allow_empty: bool = True) -> tuple[dict[str, Any], tuple[dict[str, str], int] | None]:
    """Parse a JSON body as an object with consistent error semantics."""
    if not request.is_json:
        return ({}, None) if allow_empty else ({}, ({"error": "expected application/json body"}, 400))

    try:
        data = request.get_json()
    except BadRequest as exc:
        detail = str(exc.description or "malformed json payload")
        return {}, ({"error": "invalid json payload", "detail": detail}, 400)

    if data is None:
        if allow_empty:
            return {}, None
        return {}, ({"error": "missing json body"}, 400)

    if not isinstance(data, dict):
        return {}, ({"error": "json body must be an object"}, 400)
    return data, None


def coerce_json_value(raw: Any, expected_type: type) -> tuple[Any | None, str | None]:
    """Coerce a JSON field with strict bool/int/float semantics."""
    if expected_type is bool:
        if isinstance(raw, bool):
            return raw, None
        return None, "expected boolean"

    if expected_type is int:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(raw, bool):
            return None, "expected integer"
        if isinstance(raw, int):
            return raw, None
        return None, "expected integer"

    if expected_type is float:
        if isinstance(raw, bool):
            return None, "expected number"
        if isinstance(raw, (int, float)):
            return float(raw), None
        return None, "expected number"

    if expected_type is str:
        if isinstance(raw, str):
            return raw, None
        return None, "expected string"

    try:
        return expected_type(raw), None
    except (TypeError, ValueError):
        return None, f"expected {expected_type.__name__}"
