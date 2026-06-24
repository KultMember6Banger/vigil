"""Tests for health-trend history (dependency-free)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil.history import (
    record_health_snapshot, load_history, diff_last_two, format_history,
)


def test_snapshot_appends_jsonl(tmp_path):
    p1 = record_health_snapshot(tmp_path, {"a": 1.0, "b": 0.5},
                                timestamp="2026-01-01T00:00:00",
                                totals={"files": 2, "with_issues": 1, "issues": 1})
    assert p1.name == "history.jsonl"
    record_health_snapshot(tmp_path, {"a": 0.8, "b": 0.7},
                           timestamp="2026-01-02T00:00:00")
    snaps = load_history(tmp_path)
    assert len(snaps) == 2
    assert snaps[0]["overall"] == 0.75   # (1.0 + 0.5) / 2
    assert snaps[1]["scores"]["a"] == 0.8


def test_diff_improved_and_degraded(tmp_path):
    record_health_snapshot(tmp_path, {"a": 0.5, "b": 1.0, "gone": 0.4},
                           timestamp="t1")
    record_health_snapshot(tmp_path, {"a": 0.9, "b": 0.6, "new": 0.3},
                           timestamp="t2")
    snaps = load_history(tmp_path)
    d = diff_last_two(snaps)
    improved = dict((f, (o, n)) for f, o, n in d["improved"])
    degraded = dict((f, (o, n)) for f, o, n in d["degraded"])
    assert "a" in improved          # 0.5 -> 0.9
    assert "b" in degraded          # 1.0 -> 0.6
    assert ("new", 0.3) in d["added"]
    assert ("gone", 0.4) in d["removed"]


def test_diff_single_snapshot_empty(tmp_path):
    record_health_snapshot(tmp_path, {"a": 1.0}, timestamp="t1")
    d = diff_last_two(load_history(tmp_path))
    assert d["improved"] == [] and d["degraded"] == []
    assert d["overall_delta"] == 0.0


def test_format_history_empty(tmp_path):
    out = format_history(load_history(tmp_path))
    assert "No history yet" in out


def test_format_history_renders(tmp_path):
    record_health_snapshot(tmp_path, {"a": 1.0, "b": 0.4}, timestamp="t1")
    record_health_snapshot(tmp_path, {"a": 1.0, "b": 0.9}, timestamp="t2")
    out = format_history(load_history(tmp_path))
    assert "Health Trend" in out
    assert "IMPROVED" in out
    assert "b" in out


def test_load_skips_malformed_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps({"timestamp": "t1", "overall": 1.0, "scores": {}}) +
                    "\nnot json\n" +
                    json.dumps({"timestamp": "t2", "overall": 0.9, "scores": {}}) + "\n")
    snaps = load_history(tmp_path)
    assert len(snaps) == 2
