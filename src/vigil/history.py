"""Health-trend history — append per-run snapshots and show the trend.

Every `vigil health` run appends one JSON line to `<store>/history.jsonl`:

    {"timestamp": "2026-06-24T12:00:00", "overall": 0.91,
     "totals": {"files": 10, "with_issues": 3, "issues": 7},
     "scores": {"alpha": 1.0, "beta": 0.7, ...}}

`vigil history` reads the file and reports overall health over time plus which
files improved / degraded since the previous run. Dependency-free: json + the
standard library only. The timestamp is passed in (never datetime.now() at
import time) so callers control it and tests stay deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

HISTORY_FILE = 'history.jsonl'


def record_health_snapshot(
    store_dir: Path,
    scores: dict[str, float],
    timestamp: str,
    totals: dict | None = None,
) -> Path:
    """Append a health snapshot to `<store_dir>/history.jsonl`.

    `scores` is {file_stem: health_score}; files not present are assumed 1.0 by
    the consumer. `timestamp` is an ISO string supplied by the caller. Returns
    the history file path.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / HISTORY_FILE

    rounded = {k: round(float(v), 3) for k, v in scores.items()}
    overall = round(sum(rounded.values()) / len(rounded), 3) if rounded else 1.0

    snapshot = {
        'timestamp': timestamp,
        'overall': overall,
        'totals': totals or {},
        'scores': rounded,
    }
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(snapshot) + '\n')
    return path


def load_history(store_dir: Path) -> list[dict]:
    """Load all snapshots (oldest first). Skips malformed lines."""
    path = Path(store_dir) / HISTORY_FILE
    if not path.exists():
        return []
    snapshots = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            snapshots.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return snapshots


def diff_last_two(snapshots: list[dict]) -> dict:
    """Compare the two most recent snapshots.

    Returns {improved: [(file, old, new)], degraded: [...], added: [...],
    removed: [...], overall_delta: float}. With <2 snapshots, deltas are empty.
    """
    out = {'improved': [], 'degraded': [], 'added': [], 'removed': [],
           'overall_delta': 0.0}
    if len(snapshots) < 2:
        return out

    prev, cur = snapshots[-2], snapshots[-1]
    out['overall_delta'] = round(cur.get('overall', 1.0) - prev.get('overall', 1.0), 3)

    prev_scores = prev.get('scores', {})
    cur_scores = cur.get('scores', {})

    for f, new in sorted(cur_scores.items()):
        if f not in prev_scores:
            out['added'].append((f, round(new, 3)))
            continue
        old = prev_scores[f]
        delta = round(new - old, 3)
        if delta > 0.001:
            out['improved'].append((f, round(old, 3), round(new, 3)))
        elif delta < -0.001:
            out['degraded'].append((f, round(old, 3), round(new, 3)))

    for f in sorted(prev_scores):
        if f not in cur_scores:
            out['removed'].append((f, round(prev_scores[f], 3)))

    return out


def format_history(snapshots: list[dict], max_rows: int = 20) -> str:
    """Render a textual trend report for `vigil history`."""
    if not snapshots:
        return 'No history yet. Run `vigil health <memory_dir>` to record a snapshot.'

    lines = [
        '=' * 60,
        '  VIGIL — Health Trend',
        '=' * 60,
        '',
        f'  {len(snapshots)} snapshot(s) recorded',
        '',
        '  overall health over time:',
    ]

    shown = snapshots[-max_rows:]
    for snap in shown:
        ts = snap.get('timestamp', '?')
        overall = snap.get('overall', 1.0)
        totals = snap.get('totals', {})
        bar_len = int(round(overall * 20))
        bar = '#' * bar_len + '.' * (20 - bar_len)
        extra = ''
        if totals:
            extra = (f"  ({totals.get('with_issues', '?')}/{totals.get('files', '?')}"
                     f" files w/ issues)")
        lines.append(f'    {ts}  [{bar}] {overall:.2f}{extra}')

    diff = diff_last_two(snapshots)
    lines.append('')
    if len(snapshots) >= 2:
        sign = '+' if diff['overall_delta'] >= 0 else ''
        lines.append(f'  since last run: overall {sign}{diff["overall_delta"]:+.3f}'
                     .replace('++', '+'))
        if diff['improved']:
            lines.append('')
            lines.append('  IMPROVED:')
            for f, old, new in diff['improved']:
                lines.append(f'    + {f}: {old:.2f} -> {new:.2f}')
        if diff['degraded']:
            lines.append('')
            lines.append('  DEGRADED:')
            for f, old, new in diff['degraded']:
                lines.append(f'    - {f}: {old:.2f} -> {new:.2f}')
        if diff['added']:
            lines.append('')
            lines.append('  NEW (now flagged):')
            for f, new in diff['added']:
                lines.append(f'    * {f}: {new:.2f}')
        if diff['removed']:
            lines.append('')
            lines.append('  RESOLVED (no longer flagged):')
            for f, old in diff['removed']:
                lines.append(f'    v {f}: was {old:.2f}')
        if not (diff['improved'] or diff['degraded'] or diff['added'] or diff['removed']):
            lines.append('  no per-file changes since last run.')
    else:
        lines.append('  (need 2+ snapshots to show per-file trends)')

    return '\n'.join(lines)
