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
