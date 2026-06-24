"""Tests for the shared canonical core (memcore) — Vigil copy.

The vendored copy is identical to Steno's; these mirror the dep-free helpers
(frontmatter, cosine/distance math, store/collection resolution).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil import memcore  # noqa: E402


def test_parse_frontmatter_basic():
    meta, body = memcore.parse_frontmatter('---\ntype: project\nname: svc\n---\nhello body\n')
    assert meta == {'type': 'project', 'name': 'svc'}
    assert body.strip() == 'hello body'


def test_parse_frontmatter_missing():
    meta, body = memcore.parse_frontmatter('plain text')
    assert meta == {}
    assert body == 'plain text'


def test_parse_frontmatter_non_dict_returns_empty():
    meta, _ = memcore.parse_frontmatter('---\n- a\n- b\n---\nbody')
    assert meta == {}


def test_sim_from_distance():
    assert memcore.sim_from_distance(0.0) == 1.0
    assert abs(memcore.sim_from_distance(0.2) - 0.8) < 1e-9


def test_cosine_matrix():
    embs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    m = memcore.cosine_matrix(embs)
    assert abs(m[0][1] - 1.0) < 1e-5
    assert abs(m[0][2] - 0.0) < 1e-5


def test_resolve_collection_precedence(monkeypatch):
    monkeypatch.delenv('MEMORY_COLLECTION', raising=False)
    assert memcore.resolve_collection() == 'agent_memory'
    assert memcore.resolve_collection('c') == 'c'
    monkeypatch.setenv('MEMORY_COLLECTION', 'envcol')
    assert memcore.resolve_collection() == 'envcol'


def test_resolve_store_precedence(monkeypatch):
    monkeypatch.delenv('MEMORY_STORE', raising=False)
    assert memcore.resolve_store('/x', 'MEMORY_STORE') == Path('/x')
    monkeypatch.setenv('MEMORY_STORE', '/x/env')
    assert memcore.resolve_store(None, 'MEMORY_STORE') == Path('/x/env')
    monkeypatch.delenv('MEMORY_STORE', raising=False)
    assert memcore.resolve_store(None, 'MEMORY_STORE', default='/d') == Path('/d')
    assert memcore.resolve_store(None, 'MEMORY_STORE') is None
