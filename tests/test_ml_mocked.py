"""Tests for Vigil's ML-core logic, with chromadb + sentence_transformers MOCKED.

These cover code paths that normally require ChromaDB and embedding/NLI models:
  - find_contradictions  (candidate windowing, entity-overlap filter, softmax)
  - find_duplicates      (threshold + min-text filtering)
  - find_isolated        (per-file max-similarity threshold)
  - pre_write_check      (cosine recovery sim = 1 - distance, NLI gating)
  - apply_supersession_decay (health-score decay math)
  - compute_health_scores penalties + the staleness CRITICAL threshold fix

We patch _get_all_records to return canned ids/documents/metadatas/embeddings
and patch the cached model loaders (_load_nli / _load_embedder) so nothing real
is imported. numpy IS used for real (it is a light dependency of the logic), so
these tests require numpy to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

np = pytest.importorskip("numpy")

from vigil import scanner
from vigil.scanner import (
    Issue,
    find_contradictions,
    find_duplicates,
    find_isolated,
    pre_write_check,
    apply_supersession_decay,
    compute_health_scores,
    find_stale,
    _build_sim_matrix,
    _softmax,
)


# --- canned records helpers ---

def _orthogonal_emb(idx, dim=8):
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


def _records(docs, sources, embeddings):
    return {
        "ids": [f"id{i}" for i in range(len(docs))],
        "documents": docs,
        "metadatas": [{"source_file": s} for s in sources],
        "embeddings": embeddings,
    }


# --- _softmax sanity (used by NLI scoring) ---

def test_softmax_rows_sum_to_one():
    x = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5]])
    p = _softmax(x)
    assert np.allclose(p.sum(axis=-1), 1.0)
    # higher logit -> higher prob
    assert p[0, 0] > p[0, 2]


# --- find_duplicates ---

def test_find_duplicates_threshold_and_min_text():
    long_a = "PostgreSQL is the primary database used for authentication and storage."
    long_b = "PostgreSQL is the primary database used for authentication and storage."
    short = "too short"
    docs = [long_a, long_b, short]
    sources = ["fileA", "fileB", "fileC"]
    # identical normalized embeddings for 0 and 1 -> sim 1.0; 2 orthogonal
    emb = [_orthogonal_emb(0), _orthogonal_emb(0), _orthogonal_emb(3)]
    recs = _records(docs, sources, emb)

    with mock.patch.object(scanner, "_get_all_records", return_value=recs):
        issues = find_duplicates(Path("/x"), threshold=0.85)

    assert len(issues) == 1, [i.message for i in issues]
    assert set(issues[0].files) == {"fileA", "fileB"}
    assert issues[0].details["similarity"] >= 0.99


def test_find_duplicates_same_file_skipped():
    long_a = "PostgreSQL is the primary database used for authentication and storage."
    docs = [long_a, long_a]
    sources = ["same", "same"]
    emb = [_orthogonal_emb(0), _orthogonal_emb(0)]
    recs = _records(docs, sources, emb)
    with mock.patch.object(scanner, "_get_all_records", return_value=recs):
        issues = find_duplicates(Path("/x"))
    assert issues == []


def test_find_duplicates_accepts_passed_records_and_matrix():
    # When full_scan passes records+sim_matrix, _get_all_records must NOT be called.
    long_a = "PostgreSQL is the primary database used for authentication and storage."
    docs = [long_a, long_a]
    sources = ["a", "b"]
    emb = [_orthogonal_emb(0), _orthogonal_emb(0)]
    recs = _records(docs, sources, emb)
    sm = _build_sim_matrix(emb)
    with mock.patch.object(scanner, "_get_all_records",
                           side_effect=AssertionError("should not read")):
        issues = find_duplicates(Path("/x"), records=recs, sim_matrix=sm)
    assert len(issues) == 1


# --- find_isolated ---

def test_find_isolated_flags_low_max_sim_file():
    docs = ["chunk one about alpha topic", "chunk two about alpha topic",
            "completely unrelated lonely island content"]
    sources = ["pair1", "pair2", "lonely"]
    # 0 and 1 similar; 2 orthogonal to both
    emb = [_orthogonal_emb(0), _orthogonal_emb(0), _orthogonal_emb(5)]
    recs = _records(docs, sources, emb)
    with mock.patch.object(scanner, "_get_all_records", return_value=recs):
        issues = find_isolated(Path("/x"), isolation_threshold=0.3)
    flagged = {f for i in issues for f in i.files}
    assert "lonely" in flagged
    assert "pair1" not in flagged and "pair2" not in flagged


# --- find_contradictions ---

def _make_nli(contradiction_prob):
    """Return a fake CrossEncoder whose predict yields logits that softmax to
    a given contradiction probability in slot 0 (labels [contra, neutral, entail])."""
    class FakeNLI:
        def predict(self, pairs):
            # high logit on contradiction slot
            out = []
            for _ in pairs:
                if contradiction_prob >= 0.9:
                    out.append([6.0, 0.0, 0.0])
                else:
                    out.append([0.0, 0.0, 0.0])  # uniform -> 0.33 each
            return out
    return FakeNLI()


def test_find_contradictions_entity_overlap_required():
    # Two same-topic-ish chunks with shared entity (config.md) and strong NLI.
    a = "The service must use config.md for settings and never disable it."
    b = "The service must not use config.md for settings; disable it always."
    docs = [a, b]
    sources = ["one", "two"]
    # similarity in (0.65, 0.90): build embeddings with a controlled angle
    v1 = np.array([1.0, 0.0])
    # cos = 0.8 -> angle
    import math
    ang = math.acos(0.8)
    v2 = np.array([math.cos(ang), math.sin(ang)])
    emb = [v1.tolist(), v2.tolist()]
    recs = _records(docs, sources, emb)

    with mock.patch.object(scanner, "_get_all_records", return_value=recs), \
         mock.patch.object(scanner, "_load_nli", return_value=_make_nli(0.95)):
        issues = find_contradictions(Path("/x"), nli_threshold=0.85)

    assert len(issues) == 1, [i.message for i in issues]
    assert issues[0].category == "contradiction"
    assert issues[0].severity == "CRITICAL"  # c_prob > 0.95
    assert "config.md" in issues[0].details["shared_entities"]


def test_find_contradictions_sim_window_excludes_too_similar():
    a = "Alpha config.md alpha."
    b = "Alpha config.md alpha."
    docs = [a, b]
    sources = ["one", "two"]
    emb = [_orthogonal_emb(0), _orthogonal_emb(0)]  # sim ~1.0 > sim_high -> excluded
    recs = _records(docs, sources, emb)
    with mock.patch.object(scanner, "_get_all_records", return_value=recs), \
         mock.patch.object(scanner, "_load_nli", return_value=_make_nli(0.99)):
        issues = find_contradictions(Path("/x"))
    assert issues == []


def test_find_contradictions_low_nli_filtered():
    import math
    a = "The service must use config.md and run on port 8080."
    b = "The service must not use config.md and run on port 8080."
    docs = [a, b]
    sources = ["one", "two"]
    ang = math.acos(0.8)
    emb = [[1.0, 0.0], [math.cos(ang), math.sin(ang)]]
    recs = _records(docs, sources, emb)
    with mock.patch.object(scanner, "_get_all_records", return_value=recs), \
         mock.patch.object(scanner, "_load_nli", return_value=_make_nli(0.3)):
        issues = find_contradictions(Path("/x"), nli_threshold=0.85)
    assert issues == []  # uniform softmax (~0.33) below threshold


# --- pre_write_check (cosine recovery + NLI gating) ---

class _FakeCollection:
    def __init__(self, docs, sources, distances):
        self._docs = docs
        self._sources = sources
        self._distances = distances

    def count(self):
        return len(self._docs)

    def query(self, **kwargs):
        n = len(self._docs)
        return {
            "ids": [[f"id{i}" for i in range(n)]],
            "documents": [self._docs],
            "metadatas": [[{"source_file": s} for s in self._sources]],
            "distances": [self._distances],
        }


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection

    def get_collection(self, name):
        return self._collection


class _FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.array([[1.0, 0.0]] * len(texts))


def test_pre_write_check_uses_sim_one_minus_distance():
    # distance 0.2 -> sim 0.8 (in window 0.5..0.92). With the OLD wrong formula
    # sim = 1 - 0.2/2 = 0.9 (also in window) but the correct scale matters for
    # supersession (sim > 0.65) vs contradiction gating. We assert a conflict
    # surfaces and similarity is reported as 1 - distance.
    new_text = "The database must use config.md and run on port 8080 for auth."
    existing = "The database must not use config.md and run on port 8080 for auth."
    coll = _FakeCollection([existing], ["existing_file"], [0.2])  # sim 0.8

    fake_chromadb = mock.MagicMock()
    fake_chromadb.PersistentClient.return_value = _FakeClient(coll)

    with mock.patch.dict(sys.modules, {"chromadb": fake_chromadb}), \
         mock.patch.object(scanner, "_load_embedder", return_value=_FakeEmbedder()), \
         mock.patch.object(scanner, "_load_nli", return_value=_make_nli(0.95)):
        issues = pre_write_check(new_text, Path("/x"), nli_threshold=0.7)

    assert len(issues) >= 1
    conflict = [i for i in issues if i.category == "pre_write_conflict"]
    assert conflict, [i.category for i in issues]
    assert abs(conflict[0].details["similarity"] - 0.8) < 1e-6


def test_pre_write_check_skips_out_of_window_distance():
    # distance 0.6 -> sim 0.4 < 0.5 -> skipped entirely
    new_text = "The database must use config.md and run on port 8080 for auth."
    existing = "Unrelated content about config.md and port 8080 systems here now."
    coll = _FakeCollection([existing], ["existing_file"], [0.6])

    fake_chromadb = mock.MagicMock()
    fake_chromadb.PersistentClient.return_value = _FakeClient(coll)

    with mock.patch.dict(sys.modules, {"chromadb": fake_chromadb}), \
         mock.patch.object(scanner, "_load_embedder", return_value=_FakeEmbedder()), \
         mock.patch.object(scanner, "_load_nli", return_value=_make_nli(0.95)):
        issues = pre_write_check(new_text, Path("/x"))

    assert issues == []


# --- apply_supersession_decay (health-score decay math) ---

class _DecayCollection:
    def __init__(self):
        self.updated = None
        self._records = {
            "ids": ["old:0", "old:1"],
            "metadatas": [
                {"source_file": "old", "health_score": 1.0},
                {"source_file": "old", "health_score": 0.5},
            ],
        }

    def get(self, where=None, include=None):
        return self._records

    def update(self, ids=None, metadatas=None):
        self.updated = (ids, metadatas)


def test_apply_supersession_decay_math():
    coll = _DecayCollection()
    fake_chromadb = mock.MagicMock()
    fake_chromadb.PersistentClient.return_value = _FakeClient(coll)

    issues = [Issue(severity="INFO", category="pre_write_supersession",
                    message="supersedes old", files=["old"],
                    details={"new_text": "newer version of the fact"})]

    with mock.patch.dict(sys.modules, {"chromadb": fake_chromadb}):
        n = apply_supersession_decay(issues, Path("/x"), decay_factor=0.7)

    assert n == 2
    ids, metas = coll.updated
    # 1.0 * 0.7 = 0.7 ; 0.5 * 0.7 = 0.35
    assert metas[0]["health_score"] == 0.7
    assert metas[1]["health_score"] == 0.35
    assert "superseded_by" in metas[0]


def test_apply_supersession_decay_no_supersessions():
    issues = [Issue(severity="WARNING", category="contradiction", message="x", files=["a"])]
    # no chromadb import should even be needed
    assert apply_supersession_decay(issues, Path("/x")) == 0


# --- compute_health_scores penalties ---

def test_compute_health_scores_penalties_and_floor():
    results = {
        "contradictions": [Issue("WARNING", "contradiction", "c", files=["a"])],
        "duplicates": [Issue("WARNING", "duplicate", "d", files=["a"])],
        "stale": [Issue("WARNING", "stale", "s", files=["a"],
                        details={"staleness_score": 0.8})],
        "orphans": [Issue("WARNING", "orphan", "o", files=["a"])],
        "provenance": [Issue("CRITICAL", "provenance", "p", files=["a"])],
        "isolated": [Issue("INFO", "isolated", "i", files=["a"])],
    }
    scores = compute_health_scores(results)
    # Many issues on one file -> floored at 0.1, never below.
    assert scores["a"] == 0.1

    # stale penalty scales with staleness_score
    light = compute_health_scores(
        {"stale": [Issue("WARNING", "stale", "s", files=["b"],
                         details={"staleness_score": 0.2})]})
    # 1.0 - 0.2*0.25 = 0.95
    assert abs(light["b"] - 0.95) < 1e-6


# --- staleness CRITICAL threshold fix ---

def test_find_stale_critical_threshold_fires(tmp_path):
    import os
    from datetime import datetime, timedelta

    # A very old, highly volatile file should now reach CRITICAL (staleness>0.75),
    # which was previously dead code (threshold was > 1.0).
    p = tmp_path / "volatile.md"
    p.write_text(
        "---\nname: V\ntype: project\ndescription: d\n---\n"
        "Status: pending. Currently in-progress, ongoing, active work. "
        "TODO FIXME HACK. status: blocked next pending waiting queued."
    )
    old = (datetime.now() - timedelta(days=400)).timestamp()
    os.utime(p, (old, old))

    issues = find_stale(tmp_path, warn_days=14)
    stale = [i for i in issues if "volatile" in i.files]
    assert stale, "expected a stale issue for the very old volatile file"
    assert stale[0].severity == "CRITICAL", (
        f"expected CRITICAL, got {stale[0].severity} "
        f"(score={stale[0].details['staleness_score']})"
    )


def test_find_stale_warning_below_critical(tmp_path):
    import os
    from datetime import datetime, timedelta

    p = tmp_path / "mild.md"
    # Old enough to be stale but with no volatility markers -> WARNING not CRITICAL.
    p.write_text("---\nname: M\ntype: project\ndescription: d\n---\n"
                 "Static documented knowledge that rarely changes over time.")
    old = (datetime.now() - timedelta(days=60)).timestamp()
    os.utime(p, (old, old))

    issues = find_stale(tmp_path, warn_days=14)
    mild = [i for i in issues if "mild" in i.files]
    if mild:  # may or may not cross 0.5 depending on decay; if flagged, must be WARNING
        assert mild[0].severity == "WARNING"
        assert mild[0].details["staleness_score"] <= 0.75
