"""Vigil — Memory health monitor for AI agents.

Detects contradictions, duplicates, staleness, isolated entries,
orphan references, and missing provenance in markdown-based memory stores.
"""

from __future__ import annotations

__version__ = "0.3.1"

from vigil.scanner import (
    Issue,
    find_contradictions,
    find_duplicates,
    find_isolated,
    find_orphans,
    find_stale,
    find_unprovenanced,
    pre_write_check,
    apply_supersession_decay,
    compute_health_scores,
    update_health_scores,
    full_scan,
    format_report,
)
from vigil.config import VigilConfig
from vigil.history import (
    record_health_snapshot,
    load_history,
    format_history,
    diff_last_two,
)
from vigil.fix import build_plan, apply_plan, run_fix, format_plan
