"""System resource metrics collection.

Returns a flat dict suitable for merging into a log event dict.
All fields are optional - missing values are omitted rather than set to None,
so callers can safely merge with **collect_system_metrics() without polluting
records on platforms where a metric is unavailable.
"""


def collect_system_metrics() -> dict:
    """Collect RAM, swap, VRAM, and thermal metrics.

    Safe to call on any platform. Fields present only when readable:
      ram_used_mb, ram_total_mb, ram_pct
      swap_used_mb, swap_total_mb, swap_pct
      vram_allocated_mb, vram_reserved_mb, vram_total_mb  (CUDA)
      vram_allocated_mb                                    (MPS only)
      cpu_temp_c                                           (Linux with sensors)
    """
    out = {}

    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_used_mb"]  = round(vm.used  / 1024 ** 2)
        out["ram_total_mb"] = round(vm.total / 1024 ** 2)
        out["ram_pct"]      = round(vm.percent, 1)
        sw = psutil.swap_memory()
        out["swap_used_mb"]  = round(sw.used  / 1024 ** 2)
        out["swap_total_mb"] = round(sw.total / 1024 ** 2)
        out["swap_pct"]      = round(sw.percent, 1)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            out["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 ** 2)
            out["vram_reserved_mb"]  = round(torch.cuda.memory_reserved()  / 1024 ** 2)
            props = torch.cuda.get_device_properties(0)
            out["vram_total_mb"] = round(props.total_memory / 1024 ** 2)
        elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            out["vram_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024 ** 2)
    except Exception:
        pass

    # psutil.sensors_temperatures() is Linux-only; returns None or {} on macOS/Windows.
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if temps:
            for key in ("coretemp", "cpu_thermal", "k10temp", "zenpower", "acpitz"):
                if key in temps and temps[key]:
                    out["cpu_temp_c"] = round(temps[key][0].current, 1)
                    break
    except Exception:
        pass

    return out
