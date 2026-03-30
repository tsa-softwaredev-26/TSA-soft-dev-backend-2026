"""Stress tests for settings/items/sightings churn."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.api.routes.items import items_bp
from visual_memory.api.routes.settings_route import settings_bp
from visual_memory.api.routes.sightings import sightings_bp
from visual_memory.tests.scripts.stress_artifacts import (
    append_recommendations,
    record_failure_modes,
    write_json_report,
)
from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app, seed_item

_runner = TestRunner("stress_settings_items_sightings")


def _seed(db):
    seed_item(db, "wallet", room_name="kitchen", emb_seed=1)
    seed_item(db, "keys", room_name="office", emb_seed=2)
    seed_item(db, "phone", room_name="kitchen", emb_seed=3)


client, db, cleanup = make_test_app([settings_bp, items_bp, sightings_bp], seed_fn=_seed)


def test_churn_across_routes():
    failures: list[dict] = []

    def _settings_flip(i: int) -> int:
        payload = {"projection_head_weight": 0.5 if i % 2 == 0 else 0.7}
        return client.patch("/settings", json=payload).status_code

    def _rename_cycle(i: int) -> int:
        src = "wallet" if i % 2 == 0 else "keys"
        dst = f"{src}_{i}"
        return client.post(f"/items/{src}/rename", json={"new_label": dst, "force": True}).status_code

    def _sighting_write(i: int) -> int:
        return client.post(
            "/sightings",
            json={"room_name": "In The Kitchen", "sightings": [{"label": "phone"}]},
        ).status_code

    with ThreadPoolExecutor(max_workers=12) as ex:
        settings_codes = list(ex.map(_settings_flip, range(60)))
        rename_codes = list(ex.map(_rename_cycle, range(40)))
        sighting_codes = list(ex.map(_sighting_write, range(80)))

    if any(c != 200 for c in settings_codes):
        failures.append({"case": "settings_churn", "statuses": settings_codes})
    if any(c not in (200, 404, 409) for c in rename_codes):
        failures.append({"case": "items_churn", "statuses": rename_codes})
    if any(c != 200 for c in sighting_codes):
        failures.append({"case": "sightings_churn", "statuses": sighting_codes})

    report = {
        "suite": "stress_settings_items_sightings",
        "settings_ops": len(settings_codes),
        "item_rename_ops": len(rename_codes),
        "sighting_ops": len(sighting_codes),
        "failure_count": len(failures),
    }
    write_json_report("stress_api_report.json", report)
    if failures:
        record_failure_modes(failures)
    append_recommendations(
        [
            "[stress_settings_items_sightings]",
            f"- failures: {report['failure_count']}",
            "- If rename churn causes conflicts, isolate conflict semantics and force-rename workflow.",
        ]
    )
    assert not failures, f"stress settings/items/sightings failures: {failures[:2]}"


for name, fn in [
    ("stress_settings_items_sightings:churn", test_churn_across_routes),
]:
    _runner.run(name, fn)

cleanup()
sys.exit(_runner.summary())
