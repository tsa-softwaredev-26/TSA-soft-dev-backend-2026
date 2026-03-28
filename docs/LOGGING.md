# Logging Guide

## Overview

Three log files live under `logs/` at the project root:

| File | Levels | Rotation | Retention |
|---|---|---|---|
| `app.log` | DEBUG+ (everything) | Daily | 7 days |
| `important.log` | WARNING+ only | Weekly | 60 days |
| `crash.log` | C crashes + unhandled exceptions | Never auto-deleted | Manual |

All log entries are JSON Lines format - one JSON object per line, easy to pipe into `jq` or parse programmatically.

`crash.log` is written by `faulthandler` (C-level crashes, SIGSEGV, OOM) and `sys.excepthook` (unhandled Python exceptions). It is never rotated or auto-deleted.

---

## Log Record Fields

Every record has these base fields:

| Field | Type | Description |
|---|---|---|
| `ts` | string | ISO 8601 timestamp: `2026-03-28T10:00:01` |
| `level` | string | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `module` | string | Python module name, e.g. `visual_memory.pipelines.scan_mode.pipeline` |
| `event` | string | Machine-readable event name (most records) |
| `tag` | string | Topic tag for filtering (see below) |
| `message` | string | Free-text message (some records, no event) |
| `exception` | string | Full traceback (only on exception log calls) |

### Tag values

| Tag | When used |
|---|---|
| `perf` | Pipeline timings, RAM/VRAM usage at run() boundaries |
| `vram` | VRAM layout changes (model offloads between pipeline modes) |
| `detection` | Object detection events |
| `ocr` | OCR pipeline events |
| `learning` | Projection head training events |
| `api` | API-level events |

### Performance fields (present on `tag=perf` and `tag=vram` events)

| Field | Unit | Description |
|---|---|---|
| `duration_ms` | ms | Wall time for the operation |
| `ram_used_mb` | MB | Process RSS at time of log |
| `ram_total_mb` | MB | System total RAM |
| `ram_pct` | % | RAM utilization |
| `swap_used_mb` | MB | Swap currently in use |
| `swap_total_mb` | MB | Total swap space |
| `swap_pct` | % | Swap utilization |
| `vram_allocated_mb` | MB | CUDA memory allocated by PyTorch tensors |
| `vram_reserved_mb` | MB | CUDA memory reserved by PyTorch allocator |
| `vram_total_mb` | MB | Total GPU VRAM |
| `cpu_temp_c` | C | CPU temperature (Linux only, absent on macOS) |

### Key event names

| Event | Tag | Description |
|---|---|---|
| `scan_complete` | `perf` | End of ScanPipeline.run() |
| `remember_complete` | `perf` | End of RememberPipeline.run() |
| `vram_layout` | `vram` | Model offload between pipeline modes |
| `scan_text_match` | - | Object matched in scan |
| `remember_ocr` | - | OCR result during remember |
| `remember_second_pass` | - | Detection succeeded on second-pass prompt |
| `remember_third_pass_ollama` | - | Detection succeeded via Ollama suggestion |
| `retrain_complete` | - | Projection head training finished |
| `retrain_error` | - | Training failed |

---

## logparse CLI

```bash
python -m visual_memory.utils.logparse [options]
```

### Options

```
--file PATH        Log file (default: logs/app.log)
--level LEVEL      Minimum severity: DEBUG INFO WARNING ERROR CRITICAL
--tag TAG          Filter by tag field
--module MODULE    Filter by module name (substring match)
--event EVENT      Filter by event field (substring match)
--since DURATION   Time range: 30m, 6h, 24h, 7d  or  ISO timestamp
--format FORMAT    pretty (default) | json | csv
--tail N           In pretty mode, show last N records (default: 100, 0=all)
--stats            Show aggregate statistics instead of records
--export PATH      Write output to file
```

---

## Examples

### View recent errors

```bash
python -m visual_memory.utils.logparse --level ERROR
```

### Last hour of scan performance

```bash
python -m visual_memory.utils.logparse --tag perf --event scan_complete --since 1h
```

### VRAM offload history for the last day

```bash
python -m visual_memory.utils.logparse --tag vram --since 24h
```

### All records from a specific module

```bash
python -m visual_memory.utils.logparse --module retrain
```

### Statistics summary for the last 7 days

```bash
python -m visual_memory.utils.logparse --stats --since 7d
```

Sample output:
```
=== LOG SUMMARY ===
File:    logs/app.log
Records: 4,821
Range:   2026-03-21T10:00:00 -> 2026-03-28T10:00:00

--- Levels ---
DEBUG      :      0
INFO       :  4,789
WARNING    :     28
ERROR      :      4

--- Tags ---
perf        :    712
vram        :     34
ocr         :    203

--- Events (top 20) ---
scan_text_match              :  2,341
scan_complete                :    712   avg=1.2s  p95=2.8s  max=4.1s
remember_ocr                 :    203   avg=3.4s  p95=8.1s  max=12.3s
vram_layout                  :     34
retrain_complete             :      3

--- System peaks ---
  RAM peak:           9,012 MB
  Swap peak:            512 MB
  VRAM allocated:     5,120 MB / 8,192 MB total
  VRAM reserved:      5,632 MB

--- VRAM offloads ---
  Total: 34
    scan mode:     26
    remember mode:  8
  Avg duration: 1,240 ms
```

### Export errors from the last 24 hours to CSV

```bash
python -m visual_memory.utils.logparse --level ERROR --since 24h --format csv --export errors.csv
```

### Export perf stats to a file

```bash
python -m visual_memory.utils.logparse --stats --since 7d --export weekly_stats.txt
```

### Read from important.log (WARNING+ only, 60-day history)

```bash
python -m visual_memory.utils.logparse --file logs/important.log --stats
```

### Pipe into jq for custom queries

```bash
python -m visual_memory.utils.logparse --tag perf --format json | \
  jq 'select(.event == "scan_complete") | {ts, duration_ms, match_count, vram_allocated_mb}'
```

---

## Writing log calls

```python
from visual_memory.utils import get_logger
from visual_memory.utils.logger import LogTag

_log = get_logger(__name__)

# Basic event
_log.info({"event": "my_event", "key": "value"})

# With tag for filtering
_log.info({"event": "my_event", "tag": LogTag.PERF, "duration_ms": 1234})

# Warning / error
_log.warning({"event": "threshold_exceeded", "value": 0.95, "limit": 0.9})
_log.error({"event": "ocr_failed", "reason": str(exc)})

# Exception with traceback (call inside except block)
try:
    risky()
except Exception:
    _log.exception({"event": "risky_failed"})
```

String messages also work:
```python
_log.warning("something unexpected, no structured data available")
```

---

## Querying logs directly with jq

```bash
# All ERROR records
grep '"level":"ERROR"' logs/app.log | jq .

# Scan durations over 3 seconds
cat logs/app.log | jq 'select(.event == "scan_complete" and .duration_ms > 3000)'

# VRAM high-water mark
cat logs/app.log | jq 'select(.vram_allocated_mb != null) | .vram_allocated_mb' | sort -n | tail -1
```
