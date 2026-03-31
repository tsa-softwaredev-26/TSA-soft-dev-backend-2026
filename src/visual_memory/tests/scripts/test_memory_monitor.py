from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils.memory_monitor import MemoryMonitor, _main


def test_check_memory_fields() -> None:
    monitor = MemoryMonitor()
    state = monitor.check_memory()
    for key in ["ram_pct", "swap_pct", "combined_pct", "ram_used_mb", "ram_total_mb"]:
        assert key in state
        assert isinstance(state[key], (int, float))
    assert 0.0 <= state["ram_pct"] <= 100.0
    assert 0.0 <= state["combined_pct"] <= 100.0


def test_is_oom_risk_true_when_high() -> None:
    monitor = MemoryMonitor()
    with patch.object(MemoryMonitor, "check_memory", return_value={"combined_pct": 92.0}):
        assert monitor.is_oom_risk(threshold=0.85) is True
        assert monitor.is_oom_risk(threshold=0.95) is False


def test_cleanup_zombies_skips_unowned_or_failed() -> None:
    monitor = MemoryMonitor()
    with patch.object(MemoryMonitor, "_list_zombies", return_value=[101, 102, 103]), \
         patch.object(MemoryMonitor, "_owns_process", side_effect=lambda pid: pid != 102), \
         patch.object(MemoryMonitor, "_terminate_pid", side_effect=lambda pid: pid == 101):
        killed = monitor.cleanup_zombies(max_age_hours=1)
    assert killed == [101]


def test_cleanup_zombies_does_not_touch_active_processes() -> None:
    monitor = MemoryMonitor()
    with patch.object(MemoryMonitor, "_list_zombies", return_value=[]), \
         patch.object(MemoryMonitor, "_terminate_pid") as terminate:
        killed = monitor.cleanup_zombies(max_age_hours=1)
    assert killed == []
    terminate.assert_not_called()


def test_suggest_throttle_thresholds() -> None:
    monitor = MemoryMonitor()
    with patch.object(MemoryMonitor, "check_memory", return_value={"combined_pct": 76.0, "vram_pct": 40.0}):
        assert monitor.suggest_throttle() is True
    with patch.object(MemoryMonitor, "check_memory", return_value={"combined_pct": 30.0, "vram_pct": 95.0}):
        assert monitor.suggest_throttle() is True
    with patch.object(MemoryMonitor, "check_memory", return_value={"combined_pct": 40.0, "vram_pct": 50.0}):
        assert monitor.suggest_throttle() is False


def test_cli_check_outputs_json() -> None:
    with patch("sys.argv", ["memory_monitor", "--check"]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = _main()
    assert code == 0
    parsed = json.loads(buf.getvalue())
    assert "combined_pct" in parsed


def test_cli_cleanup_prints_count() -> None:
    with patch.object(MemoryMonitor, "cleanup_zombies", return_value=[11, 22]), \
         patch("sys.argv", ["memory_monitor", "--cleanup"]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = _main()
    assert code == 0
    assert "Killed 2 zombie processes" in buf.getvalue()


def test_cli_cleanup_log_only_no_stdout() -> None:
    with patch.object(MemoryMonitor, "cleanup_zombies", return_value=[11, 22]), \
         patch("sys.argv", ["memory_monitor", "--cleanup", "--log-only"]):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = _main()
    assert code == 0
    assert buf.getvalue() == ""


if __name__ == "__main__":
    runner = TestRunner("memory_monitor")
    for name, fn in [
        ("memory_monitor:check_memory_fields", test_check_memory_fields),
        ("memory_monitor:is_oom_risk", test_is_oom_risk_true_when_high),
        ("memory_monitor:cleanup_zombies", test_cleanup_zombies_skips_unowned_or_failed),
        ("memory_monitor:cleanup_skips_active", test_cleanup_zombies_does_not_touch_active_processes),
        ("memory_monitor:suggest_throttle", test_suggest_throttle_thresholds),
        ("memory_monitor:cli_check", test_cli_check_outputs_json),
        ("memory_monitor:cli_cleanup", test_cli_cleanup_prints_count),
        ("memory_monitor:cli_cleanup_log_only", test_cli_cleanup_log_only_no_stdout),
    ]:
        runner.run(name, fn)
    raise SystemExit(runner.summary())
