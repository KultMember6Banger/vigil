"""Tests for Vigil scanner — lightweight checks only (no ChromaDB/embeddings)."""

import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil.scanner import (
    Issue,
    find_stale,
    find_orphans,
    find_unprovenanced,
    compute_health_scores,
    format_report,
    _ebbinghaus_retention,
)


def _write_memory(tmpdir: Path, name: str, content: str) -> Path:
    p = tmpdir / name
    p.write_text(content)
    return p


def test_find_unprovenanced():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _write_memory(d, "good.md", "---\nname: Good\ntype: feedback\ndescription: Has all fields\n---\nContent here.")
        _write_memory(d, "missing_type.md", "---\nname: Missing Type\ndescription: No type field\n---\nSome content.")
        _write_memory(d, "no_frontmatter.md", "Just plain text, no metadata at all.")

        issues = find_unprovenanced(d)
        names = [f for i in issues for f in i.files]
        assert any("missing_type" in f for f in names), f"Should flag missing_type.md: {names}"
        assert any("no_frontmatter" in f for f in names), f"Should flag no_frontmatter.md: {names}"
        assert not any("good" in f for f in names), f"Should not flag good.md: {names}"
        print(f"  OK find_unprovenanced ({len(issues)} issues)")


def test_find_orphans():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        _write_memory(d, "refs.md", "---\nname: References\ntype: reference\ndescription: Has refs\n---\nSee [other](other.md) and [broken](nonexistent.md).")
        _write_memory(d, "other.md", "---\nname: Other\ntype: project\ndescription: Exists\n---\nContent.")

        issues = find_orphans(d)
        broken = [i for i in issues if "nonexistent" in i.message]
        assert len(broken) > 0, f"Should find broken ref to nonexistent.md: {[i.message for i in issues]}"
        print(f"  OK find_orphans ({len(issues)} issues)")


def test_find_stale():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        p = _write_memory(d, "old.md", "---\nname: Old Memory\ntype: project\ndescription: Stale\n---\nStatus: pending review. TODO fix this.")
        import os
        old_time = (datetime.now(timezone.utc) - timedelta(days=90)).timestamp()
        os.utime(p, (old_time, old_time))

        _write_memory(d, "new.md", "---\nname: Fresh\ntype: feedback\ndescription: Recent\n---\nStatic knowledge that doesn't change.")

        issues = find_stale(d, warn_days=30)
        stale_files = [f for i in issues for f in i.files]
        assert any("old" in f for f in stale_files), f"Should flag 90-day-old file: {stale_files}"
        print(f"  OK find_stale ({len(issues)} issues)")


def test_ebbinghaus_retention():
    r0 = _ebbinghaus_retention(0)
    assert abs(r0 - 1.0) < 0.01, f"Day 0 retention should be ~1.0, got {r0}"

    r30 = _ebbinghaus_retention(30)
    r90 = _ebbinghaus_retention(90)
    assert r30 > r90, f"30-day retention ({r30}) should exceed 90-day ({r90})"

    r30_strong = _ebbinghaus_retention(30, strength=3.0)
    assert r30_strong > r30, f"Stronger memory ({r30_strong}) should retain better than default ({r30})"
    print(f"  OK ebbinghaus_retention (30d={r30:.2f}, 90d={r90:.2f}, 30d-strong={r30_strong:.2f})")


def test_compute_health_scores():
    results = {
        "stale": [Issue(severity="WARNING", category="stale", message="old", files=["a.md"])],
        "orphans": [Issue(severity="WARNING", category="orphan", message="broken ref", files=["a.md", "b.md"])],
        "provenance": [Issue(severity="INFO", category="provenance", message="missing type", files=["c.md"])],
    }
    scores = compute_health_scores(results)
    assert scores["a.md"] < 1.0, f"a.md has 2 issues, should be < 1.0: {scores['a.md']}"
    assert scores["b.md"] < 1.0, f"b.md has 1 issue, should be < 1.0: {scores['b.md']}"
    assert scores["a.md"] < scores["b.md"], "a.md (2 issues) should score lower than b.md (1 issue)"
    print(f"  OK compute_health_scores (a={scores['a.md']:.2f}, b={scores['b.md']:.2f})")


def test_format_report():
    results = {
        "stale": [Issue(severity="WARNING", category="stale", message="90 days old", files=["old.md"])],
        "provenance": [],
    }
    report = format_report(results)
    assert "90 days old" in report, f"Report should contain issue message"
    assert "old.md" in report, f"Report should contain filename"
    print(f"  OK format_report ({len(report)} chars)")


if __name__ == "__main__":
    print("Vigil scanner tests:")
    test_find_unprovenanced()
    test_find_orphans()
    test_find_stale()
    test_ebbinghaus_retention()
    test_compute_health_scores()
    test_format_report()
    print("\nAll tests passed.")
