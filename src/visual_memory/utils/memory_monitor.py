from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from visual_memory.utils.logger import LogTag, get_logger

_logger = get_logger(__name__)
_HZ = os.sysconf(os.sysconf_names["SC_CLK_TCK"]) if hasattr(os, "sysconf") else 100


@dataclass
class MemoryState:
    ram_pct: float
    swap_pct: float
    combined_pct: float
    ram_used_mb: float
    ram_total_mb: float
    swap_used_mb: float
    swap_total_mb: float
    vram_pct: Optional[float]
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]

    def as_dict(self) -> dict:
        return {
            "ram_pct": round(self.ram_pct, 2),
            "swap_pct": round(self.swap_pct, 2),
            "combined_pct": round(self.combined_pct, 2),
            "ram_used_mb": round(self.ram_used_mb, 2),
            "ram_total_mb": round(self.ram_total_mb, 2),
            "swap_used_mb": round(self.swap_used_mb, 2),
            "swap_total_mb": round(self.swap_total_mb, 2),
            "vram_pct": round(self.vram_pct, 2) if self.vram_pct is not None else None,
            "vram_used_mb": round(self.vram_used_mb, 2) if self.vram_used_mb is not None else None,
            "vram_total_mb": round(self.vram_total_mb, 2) if self.vram_total_mb is not None else None,
        }


class MemoryMonitor:
    def __init__(self) -> None:
        self._uid = os.getuid() if hasattr(os, "getuid") else None

    def check_memory(self) -> dict:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        ram_used_mb = vm.used / (1024 ** 2)
        ram_total_mb = vm.total / (1024 ** 2)
        swap_used_mb = sw.used / (1024 ** 2)
        swap_total_mb = sw.total / (1024 ** 2)

        denom = ram_total_mb + swap_total_mb
        combined_pct = ((ram_used_mb + swap_used_mb) / denom * 100.0) if denom > 0 else vm.percent

        vram_used_mb, vram_total_mb = self._read_vram_mb()
        vram_pct = None
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0:
            vram_pct = (vram_used_mb / vram_total_mb) * 100.0

        state = MemoryState(
            ram_pct=float(vm.percent),
            swap_pct=float(sw.percent),
            combined_pct=float(combined_pct),
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            swap_used_mb=swap_used_mb,
            swap_total_mb=swap_total_mb,
            vram_pct=vram_pct,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
        )
        return state.as_dict()

    def is_oom_risk(self, threshold: float = 0.85) -> bool:
        state = self.check_memory()
        return (state["combined_pct"] / 100.0) >= threshold

    def suggest_throttle(self) -> bool:
        state = self.check_memory()
        combined = state["combined_pct"] / 100.0
        if combined >= 0.75:
            return True
        vram_pct = state.get("vram_pct")
        return bool(vram_pct is not None and (vram_pct / 100.0) >= 0.9)

    def log_memory_state(self, level: str = "warning") -> None:
        state = self.check_memory()
        payload = {
            "event": "memory_state",
            "tag": LogTag.VRAM,
            "severity": level,
            **state,
        }
        if level == "critical":
            _logger.critical(payload)
        else:
            _logger.warning(payload)

    def cleanup_zombies(self, max_age_hours: float = 2) -> list[int]:
        killed: list[int] = []
        for pid in self._list_zombies(max_age_hours=max_age_hours):
            if self._uid is not None and not self._owns_process(pid):
                continue
            if self._terminate_pid(pid):
                killed.append(pid)
        if killed:
            _logger.warning(
                {
                    "event": "memory_cleanup",
                    "tag": LogTag.VRAM,
                    "killed_pids": killed,
                    "count": len(killed),
                }
            )
        return killed

    def _list_zombies(self, max_age_hours: float) -> list[int]:
        out: list[int] = []
        if os.name != "posix":
            return out
        proc_root = Path("/proc")
        if not proc_root.exists():
            return out
        now = time.time()
        boot = psutil.boot_time()
        for proc_dir in proc_root.iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            stat = self._read_proc_stat(proc_dir / "stat")
            if stat is None:
                continue
            state, start_time = stat
            if state != "Z":
                continue
            started_at = boot + (start_time / _HZ)
            age_hours = (now - started_at) / 3600.0
            if age_hours >= max_age_hours:
                out.append(pid)
        return out

    @staticmethod
    def _read_proc_stat(stat_path: Path) -> Optional[tuple[str, int]]:
        try:
            text = stat_path.read_text(encoding="utf-8")
        except Exception:
            return None
        rparen = text.rfind(")")
        if rparen == -1:
            return None
        tail = text[rparen + 2 :].split()
        if len(tail) < 20:
            return None
        state = tail[0]
        try:
            start_time = int(tail[19])
        except Exception:
            return None
        return state, start_time

    def _owns_process(self, pid: int) -> bool:
        try:
            return os.stat(f"/proc/{pid}").st_uid == self._uid
        except Exception:
            return False

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate_pid(self, pid: int) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._pid_exists(pid):
                return True
            time.sleep(0.2)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not self._pid_exists(pid):
                return True
            time.sleep(0.2)
        return not self._pid_exists(pid)

    @staticmethod
    def _read_vram_mb() -> tuple[Optional[float], Optional[float]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
        except Exception:
            return None, None

        lines = proc.stdout.strip().splitlines()
        if not lines:
            return None, None
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) != 2:
            return None, None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None, None


def _main() -> int:
    parser = argparse.ArgumentParser(description="Memory monitor utilities")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--max-age", type=float, default=2.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--log-only", action="store_true")
    args = parser.parse_args()

    monitor = MemoryMonitor()
    if args.check:
        print(json.dumps(monitor.check_memory(), indent=2))
    if args.cleanup:
        pids = monitor.cleanup_zombies(max_age_hours=args.max_age)
        if not args.log_only:
            print(f"Killed {len(pids)} zombie processes: {pids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
