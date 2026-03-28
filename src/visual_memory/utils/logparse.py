"""Log filter and statistics CLI for VisualMemory.

Usage:
    python -m visual_memory.utils.logparse [options]

See docs/LOGGING.md for a full tutorial and examples.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

_DEFAULT_LOG = Path(__file__).resolve().parents[3] / "logs" / "app.log"

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

_COL_TS     = 19
_COL_LEVEL  = 8
_COL_MODULE = 32
_COL_EVENT  = 26


def _parse_since(s: str) -> datetime:
    """Parse a relative duration string or ISO timestamp into a datetime cutoff."""
    s = s.strip()
    if s[-1] == "m" and s[:-1].isdigit():
        return datetime.now() - timedelta(minutes=int(s[:-1]))
    if s[-1] == "h" and s[:-1].isdigit():
        return datetime.now() - timedelta(hours=int(s[:-1]))
    if s[-1] == "d" and s[:-1].isdigit():
        return datetime.now() - timedelta(days=int(s[:-1]))
    return datetime.fromisoformat(s)


def _load_records(
    path: Path,
    since: datetime | None,
    min_level: str | None,
    tag: str | None,
    module: str | None,
    event: str | None,
) -> list[dict]:
    if not path.exists():
        print(f"Log file not found: {path}", file=sys.stderr)
        return []

    min_no = _LEVEL_ORDER.get((min_level or "").upper(), 0)
    records = []

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            if since is not None:
                ts_str = r.get("ts", "")
                try:
                    if datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S") < since:
                        continue
                except ValueError:
                    continue

            if min_level and _LEVEL_ORDER.get(r.get("level", ""), 0) < min_no:
                continue
            if tag and r.get("tag") != tag:
                continue
            if module and module.lower() not in r.get("module", "").lower():
                continue
            if event and event.lower() not in r.get("event", "").lower():
                continue

            records.append(r)

    return records


def _fmt_pretty(records: list[dict], tail: int | None) -> str:
    shown = records[-tail:] if tail and len(records) > tail else records
    lines = []
    for r in shown:
        ts     = r.get("ts", "")[:_COL_TS].ljust(_COL_TS)
        level  = r.get("level", "")[:_COL_LEVEL].ljust(_COL_LEVEL)
        module = r.get("module", "")
        if len(module) > _COL_MODULE:
            module = "..." + module[-((_COL_MODULE - 3)):]
        module = module.ljust(_COL_MODULE)
        evt    = r.get("event", r.get("message", ""))[:_COL_EVENT].ljust(_COL_EVENT)
        skip   = {"ts", "level", "module", "event", "message", "exception"}
        extras = "  ".join(
            f"{k}={v}" for k, v in r.items()
            if k not in skip and v is not None
        )
        exc = r.get("exception", "")
        if exc:
            extras += f"  exception={exc[:120].replace(chr(10), ' ')}"
        lines.append(f"{ts}  {level}  {module}  {evt}  {extras}")
    if tail and len(records) > tail:
        lines.insert(0, f"(showing last {tail} of {len(records)} matching records)\n")
    return "\n".join(lines)


def _fmt_csv(records: list[dict]) -> str:
    if not records:
        return ""
    all_keys: list[str] = []
    seen: set = set()
    priority = ["ts", "level", "module", "event", "tag", "duration_ms", "message"]
    for k in priority:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow({k: r.get(k, "") for k in all_keys})
    return buf.getvalue()


def _fmt_json(records: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in records)


def _p(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(int(len(values) * pct), len(values) - 1)
    return sorted(values)[idx]


def _compute_stats(records: list[dict]) -> dict:
    level_counts: Counter = Counter()
    tag_counts:   Counter = Counter()
    event_counts: Counter = Counter()
    event_durations: dict[str, list[float]] = defaultdict(list)
    ram_used:   list[float] = []
    swap_used:  list[float] = []
    vram_alloc: list[float] = []
    vram_res:   list[float] = []
    vram_total: list[float] = []
    offload_modes: Counter  = Counter()
    offload_durations: list[float] = []
    ts_list: list[str] = []

    for r in records:
        level = r.get("level", "UNKNOWN")
        level_counts[level] += 1

        tag = r.get("tag")
        if tag:
            tag_counts[tag] += 1

        evt = r.get("event", "")
        if evt:
            event_counts[evt] += 1
            dur = r.get("duration_ms")
            if dur is not None:
                try:
                    event_durations[evt].append(float(dur))
                except (TypeError, ValueError):
                    pass

        if r.get("ram_used_mb") is not None:
            try:
                ram_used.append(float(r["ram_used_mb"]))
            except (TypeError, ValueError):
                pass
        if r.get("swap_used_mb") is not None:
            try:
                swap_used.append(float(r["swap_used_mb"]))
            except (TypeError, ValueError):
                pass
        if r.get("vram_allocated_mb") is not None:
            try:
                vram_alloc.append(float(r["vram_allocated_mb"]))
            except (TypeError, ValueError):
                pass
        if r.get("vram_reserved_mb") is not None:
            try:
                vram_res.append(float(r["vram_reserved_mb"]))
            except (TypeError, ValueError):
                pass
        if r.get("vram_total_mb") is not None:
            try:
                vram_total.append(float(r["vram_total_mb"]))
            except (TypeError, ValueError):
                pass

        if evt == "vram_layout":
            mode = r.get("mode", "unknown")
            offload_modes[mode] += 1
            dur = r.get("duration_ms")
            if dur is not None:
                try:
                    offload_durations.append(float(dur))
                except (TypeError, ValueError):
                    pass

        ts = r.get("ts")
        if ts:
            ts_list.append(ts)

    return {
        "total":           len(records),
        "ts_min":          min(ts_list) if ts_list else None,
        "ts_max":          max(ts_list) if ts_list else None,
        "level_counts":    dict(level_counts),
        "tag_counts":      dict(tag_counts),
        "event_counts":    dict(event_counts.most_common(20)),
        "event_durations": {
            evt: {
                "count": len(v),
                "avg_ms":  round(sum(v) / len(v)),
                "p95_ms":  round(_p(v, 0.95)),
                "max_ms":  round(max(v)),
            }
            for evt, v in event_durations.items() if len(v) >= 2
        },
        "ram_peak_mb":      round(max(ram_used))  if ram_used  else None,
        "swap_peak_mb":     round(max(swap_used)) if swap_used else None,
        "vram_alloc_peak_mb": round(max(vram_alloc)) if vram_alloc else None,
        "vram_res_peak_mb":   round(max(vram_res))   if vram_res   else None,
        "vram_total_mb":    round(max(vram_total)) if vram_total else None,
        "offload_count":    sum(offload_modes.values()),
        "offload_by_mode":  dict(offload_modes),
        "offload_avg_ms":   round(sum(offload_durations) / len(offload_durations)) if offload_durations else None,
    }


def _fmt_stats(stats: dict, file_path: Path) -> str:
    lines = ["=== LOG SUMMARY ==="]
    lines.append(f"File:    {file_path}")
    lines.append(f"Records: {stats['total']:,}")
    if stats["ts_min"]:
        lines.append(f"Range:   {stats['ts_min']} -> {stats['ts_max']}")

    lines.append("")
    lines.append("--- Levels ---")
    for lvl in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        cnt = stats["level_counts"].get(lvl, 0)
        if cnt or lvl in ("INFO", "WARNING", "ERROR"):
            lines.append(f"  {lvl:<10}: {cnt:>6,}")

    if stats["tag_counts"]:
        lines.append("")
        lines.append("--- Tags ---")
        for tag, cnt in sorted(stats["tag_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"  {tag:<12}: {cnt:>6,}")

    lines.append("")
    lines.append("--- Events (top 20) ---")
    dur_data = stats["event_durations"]
    for evt, cnt in stats["event_counts"].items():
        d = dur_data.get(evt)
        if d:
            dur_str = f"   avg={d['avg_ms']/1000:.1f}s  p95={d['p95_ms']/1000:.1f}s  max={d['max_ms']/1000:.1f}s"
        else:
            dur_str = ""
        lines.append(f"  {evt:<30}: {cnt:>6,}{dur_str}")

    peaks = [
        stats["ram_peak_mb"],
        stats["vram_alloc_peak_mb"],
        stats["swap_peak_mb"],
    ]
    if any(p is not None for p in peaks):
        lines.append("")
        lines.append("--- System peaks ---")
        if stats["ram_peak_mb"] is not None:
            lines.append(f"  RAM peak:         {stats['ram_peak_mb']:>6,} MB")
        if stats["swap_peak_mb"] is not None:
            lines.append(f"  Swap peak:        {stats['swap_peak_mb']:>6,} MB")
        if stats["vram_alloc_peak_mb"] is not None:
            vram_total = stats["vram_total_mb"]
            total_str = f" / {vram_total:,} MB total" if vram_total else ""
            lines.append(f"  VRAM allocated:   {stats['vram_alloc_peak_mb']:>6,} MB{total_str}")
        if stats["vram_res_peak_mb"] is not None:
            lines.append(f"  VRAM reserved:    {stats['vram_res_peak_mb']:>6,} MB")

    if stats["offload_count"]:
        lines.append("")
        lines.append("--- VRAM offloads ---")
        lines.append(f"  Total: {stats['offload_count']}")
        for mode, cnt in stats["offload_by_mode"].items():
            lines.append(f"    {mode:<16}: {cnt}")
        if stats["offload_avg_ms"] is not None:
            lines.append(f"  Avg duration: {stats['offload_avg_ms']:,} ms")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m visual_memory.utils.logparse",
        description="Filter and analyze VisualMemory log files. See docs/LOGGING.md for examples.",
    )
    parser.add_argument(
        "--file", type=Path, default=_DEFAULT_LOG,
        help="Log file to parse (default: logs/app.log). Use logs/important.log for WARNING+ only.",
    )
    parser.add_argument(
        "--level", metavar="LEVEL",
        help="Minimum severity: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--tag", metavar="TAG",
        help="Filter by tag field (perf, detection, learning, api, ocr, vram)",
    )
    parser.add_argument(
        "--module", metavar="MODULE",
        help="Filter by module name substring (e.g. scan_mode, retrain)",
    )
    parser.add_argument(
        "--event", metavar="EVENT",
        help="Filter by event field substring (e.g. scan_complete, vram_layout)",
    )
    parser.add_argument(
        "--since", metavar="DURATION",
        help="Only include records from last N time units: 30m, 6h, 24h, 7d. "
             "Or an ISO timestamp: 2026-03-28T10:00:00",
    )
    parser.add_argument(
        "--format", choices=["pretty", "json", "csv"], default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--tail", type=int, default=100, metavar="N",
        help="In pretty mode, show last N matching records (default: 100). 0 = all.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show aggregate statistics instead of individual records",
    )
    parser.add_argument(
        "--export", type=Path, metavar="PATH",
        help="Write output to a file instead of (or in addition to) stdout",
    )

    args = parser.parse_args()

    since = None
    if args.since:
        try:
            since = _parse_since(args.since)
        except (ValueError, IndexError):
            print(f"Cannot parse --since value: {args.since!r}", file=sys.stderr)
            sys.exit(1)

    records = _load_records(
        path=args.file,
        since=since,
        min_level=args.level,
        tag=args.tag,
        module=args.module,
        event=args.event,
    )

    if args.stats:
        output = _fmt_stats(_compute_stats(records), args.file)
    elif args.format == "pretty":
        tail = args.tail if args.tail > 0 else None
        output = _fmt_pretty(records, tail)
    elif args.format == "csv":
        output = _fmt_csv(records)
    else:
        output = _fmt_json(records)

    print(output)

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(output, encoding="utf-8")
        print(f"\nExported to {args.export}", file=sys.stderr)


if __name__ == "__main__":
    main()
