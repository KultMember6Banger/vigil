"""Tests for `vigil fix` — resolution plan + archive (no heavy deps).

build_plan reads access/health from ChromaDB; we mock those helpers so the
planning + archiving logic runs without a real store. archive_file uses the real
filesystem (small temp dirs).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil import fix as fixmod
from vigil.fix import (
    build_plan, apply_plan, archive_file, format_plan, plan_to_dict,
)
from vigil.config import VigilConfig
from vigil.scanner import Issue


def _touch(d: Path, stem: str, content: str = "x") -> Path:
    p = d / f"{stem}.md"
    p.write_text(content)
    return p


def test_archive_file_moves_not_deletes(tmp_path):
    _touch(tmp_path, "old")
    dest = archive_file(tmp_path, "old", "archive")
    assert dest is not None
    assert not (tmp_path / "old.md").exists()        # moved away
    assert dest.exists()                              # exists in archive
    assert dest.parent.name == "archive"


def test_archive_file_collision_suffix(tmp_path):
    _touch(tmp_path, "dup")
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "dup.md").write_text("existing")
    dest = archive_file(tmp_path, "dup", "archive")
    assert dest.name == "dup.1.md"
    assert (tmp_path / "archive" / "dup.md").read_text() == "existing"


def test_archive_missing_file_returns_none(tmp_path):
    assert archive_file(tmp_path, "nope", "archive") is None


def test_build_plan_stale_archives(tmp_path):
    _touch(tmp_path, "stale1")
    results = {"stale": [Issue("WARNING", "stale", "old", files=["stale1"],
                               details={"staleness_score": 0.8})]}
    cfg = VigilConfig.defaults()
    with mock.patch("vigil.scanner._get_access_data", return_value={}), \
         mock.patch("vigil.resolve.load_health_data", return_value={}):
        actions = build_plan(results, tmp_path, tmp_path / ".vigil", cfg)
    stale_actions = [a for a in actions if a.category == "stale"]
    assert len(stale_actions) == 1
    assert stale_actions[0].action == "archive"
    assert stale_actions[0].target == "stale1"


def test_build_plan_duplicate_conservative_threshold(tmp_path):
    _touch(tmp_path, "keep", "x" * 100)
    _touch(tmp_path, "loser", "x" * 100)
    cfg = VigilConfig.defaults()  # fix_duplicate_threshold = 0.92

    # below threshold -> review only
    low = {"duplicates": [Issue("WARNING", "duplicate", "near", files=["keep", "loser"],
                                details={"similarity": 0.88})]}
    with mock.patch("vigil.scanner._get_access_data", return_value={}), \
         mock.patch("vigil.resolve.load_health_data",
                    return_value={"keep": 1.0, "loser": 0.5}):
        actions = build_plan(low, tmp_path, tmp_path / ".vigil", cfg)
    dup = [a for a in actions if a.category == "duplicates"][0]
    assert dup.action == "review"
    assert dup.details["keep"] == "keep"
    assert dup.details["archive"] == "loser"

    # at/above threshold -> archive the loser
    high = {"duplicates": [Issue("WARNING", "duplicate", "near", files=["keep", "loser"],
                                 details={"similarity": 0.95})]}
    with mock.patch("vigil.scanner._get_access_data", return_value={}), \
         mock.patch("vigil.resolve.load_health_data",
                    return_value={"keep": 1.0, "loser": 0.5}):
        actions = build_plan(high, tmp_path, tmp_path / ".vigil", cfg)
    dup = [a for a in actions if a.category == "duplicates"][0]
    assert dup.action == "archive"
    assert dup.target == "loser"


def test_build_plan_orphans_and_provenance_advisory(tmp_path):
    results = {
        "orphans": [Issue("WARNING", "orphan", "broken", files=["a"],
                          details={"missing_ref": "ghost.md", "line": 7})],
        "provenance": [Issue("CRITICAL", "provenance", "missing", files=["b"],
                            details={"missing_fields": ["type", "description"]})],
        "contradictions": [Issue("CRITICAL", "contradiction", "conflict",
                                files=["c", "d"],
                                details={"suggested_keep": "c",
                                         "suggested_archive": "d"})],
    }
    cfg = VigilConfig.defaults()
    with mock.patch("vigil.scanner._get_access_data", return_value={}), \
         mock.patch("vigil.resolve.load_health_data", return_value={}):
        actions = build_plan(results, tmp_path, tmp_path / ".vigil", cfg)
    cats = {a.category: a for a in actions}
    assert cats["orphans"].action == "manual_edit"
    assert ":7" in cats["orphans"].reason
    assert cats["provenance"].action == "add_metadata"
    assert cats["contradictions"].action == "review"   # advisory, never archive
    assert cats["contradictions"].details["suggested_keep"] == "c"
    # No contradiction/orphan/provenance action ever archives.
    assert all(a.action != "archive"
               for a in actions
               if a.category in ("contradictions", "orphans", "provenance"))


def test_apply_plan_only_archives(tmp_path):
    _touch(tmp_path, "s")
    actions = [
        fixmod.Action("stale", "archive", "s", "old"),
        fixmod.Action("contradictions", "review", "c", "advisory"),
    ]
    cfg = VigilConfig.defaults()
    apply_plan(actions, tmp_path, cfg)
    assert actions[0].applied is True
    assert "archived_to" in actions[0].details
    assert actions[1].applied is False
    assert not (tmp_path / "s.md").exists()


def test_plan_to_dict_and_format(tmp_path):
    actions = [fixmod.Action("stale", "archive", "s", "old")]
    d = plan_to_dict(actions, applied=False)
    assert d["summary"]["archive_actions"] == 1
    assert d["summary"]["archived"] == 0
    text = format_plan(actions, applied=False, archive_dir_name="archive")
    assert "DRY RUN" in text
    assert "STALE" in text
