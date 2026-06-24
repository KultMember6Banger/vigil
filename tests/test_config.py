"""Tests for `.vigil.toml` config loading (lightweight — no heavy deps)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil.config import (
    VigilConfig, parse_toml, _fallback_parse_toml, _unescape_basic,
)


EXAMPLE = (Path(__file__).resolve().parent.parent / ".vigil.toml.example")


def test_defaults_complete():
    cfg = VigilConfig.defaults()
    assert cfg.sim_low == 0.65
    assert cfg.sim_high == 0.90
    assert cfg.duplicate_threshold == 0.85
    assert cfg.fix_duplicate_threshold == 0.92
    assert cfg.archive_dir == "archive"
    assert cfg.required_provenance_fields == ["name", "type", "description"]
    assert len(cfg.volatility_markers) == 4


def test_fallback_parses_sections_and_arrays():
    text = (
        'required_provenance_fields = ["name", "type"]\n'
        'ignore_globs = ["scratch/*", "*.draft.md"]\n'
        '[contradictions]\n'
        'sim_low = 0.5\n'
        'nli_threshold = 0.9\n'
        '[fix]\n'
        'archive_dir = "trash"\n'
        'duplicate_threshold = 0.95\n'
    )
    d = _fallback_parse_toml(text)
    assert d["required_provenance_fields"] == ["name", "type"]
    assert d["ignore_globs"] == ["scratch/*", "*.draft.md"]
    assert d["contradictions"]["sim_low"] == 0.5
    assert d["fix"]["archive_dir"] == "trash"


def test_fallback_array_of_tables():
    text = (
        "[staleness]\n"
        "warn_days = 7\n"
        "[[staleness.volatility_markers]]\n"
        'pattern = "\\\\bfoo\\\\b"\n'
        'label = "foo"\n'
        "weight = 0.4\n"
        "[[staleness.volatility_markers]]\n"
        'pattern = "bar"\n'
        'label = "bar"\n'
        "weight = 0.1\n"
    )
    d = _fallback_parse_toml(text)
    markers = d["staleness"]["volatility_markers"]
    assert len(markers) == 2
    assert markers[0]["label"] == "foo"
    assert markers[1]["weight"] == 0.1


def test_basic_string_escape_yields_working_regex():
    # `\\b` in a TOML basic string is a literal backslash + b -> a valid \b regex.
    val = _unescape_basic("\\\\b(current)\\\\b".replace("\\\\", "\\"))
    # construct via the parser to mirror real loading
    d = _fallback_parse_toml('p = "\\\\bcurrent\\\\b"')
    pat = d["p"]
    assert re.search(pat, "it is current now"), repr(pat)


def test_example_file_loads():
    d = parse_toml(EXAMPLE.read_text())
    cfg = VigilConfig.from_dict(d, source_path=str(EXAMPLE))
    assert cfg.fix_duplicate_threshold == 0.92
    assert len(cfg.volatility_markers) == 4
    # every marker pattern must compile
    for m in cfg.volatility_markers:
        re.compile(m["pattern"])


def test_cli_override_ignores_none():
    cfg = VigilConfig.defaults()
    cfg.override(sim_low=0.7, sim_high=None)
    assert cfg.sim_low == 0.7
    assert cfg.sim_high == 0.90  # None override left default intact


def test_load_missing_file_is_defaults(tmp_path):
    cfg = VigilConfig.load(tmp_path)
    assert cfg.sim_low == 0.65
    assert cfg.source_path == ""


def test_load_from_memory_dir(tmp_path):
    (tmp_path / ".vigil.toml").write_text(
        "[duplicates]\nthreshold = 0.99\n")
    cfg = VigilConfig.load(tmp_path)
    assert cfg.duplicate_threshold == 0.99
    assert cfg.source_path.endswith(".vigil.toml")
