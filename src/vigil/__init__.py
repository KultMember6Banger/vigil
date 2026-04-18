"""Vigil — Memory health monitor for AI agents.

Detects contradictions, duplicates, staleness, isolated entries,
orphan references, and missing provenance in markdown-based memory stores.
"""

__version__ = "0.1.0"

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
