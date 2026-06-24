"""Canonical shared memory core for the Steno + Vigil ecosystem.

Canonical shared core — keep both copies in sync; a published `memcore`
package is the eventual home. The IDENTICAL file is vendored into both repos
(steno/memcore.py and vigil/src/vigil/memcore.py) so neither depends on the
other at import time; only the import path differs per repo.

This module centralises the obviously-duplicated plumbing both repos
re-implemented independently:
  - YAML frontmatter parsing
  - a cached SentenceTransformer loader
  - the ChromaDB cosine-distance -> similarity conversion (and a cosine matrix)
  - PersistentClient + get_or_create_collection(hnsw cosine) setup
  - store-dir / collection-name resolution honoring MEMORY_STORE /
    MEMORY_COLLECTION (plus repo-specific fallback env vars).

Heavy deps (chromadb, sentence-transformers, numpy) are imported lazily inside
the functions that need them, so importing memcore is cheap and side-effect
free even where those deps are absent.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import yaml

# Shared defaults — both repos point at the same collection by default so the
# auditor (Vigil) can read exactly what the runtime (Steno) writes.
DEFAULT_COLLECTION_NAME = 'agent_memory'
DEFAULT_EMBED_MODEL = 'all-MiniLM-L6-v2'

# Leading --- ... --- YAML block. Identical shape in both repos.
FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Uses yaml.safe_load so nested YAML, lists, and multiline values parse
    correctly. Returns (metadata_dict, body_text). On missing or invalid
    frontmatter, returns ({}, full_text).

    NOTE: the body is returned verbatim (NOT stripped). Callers that want a
    stripped body should strip it themselves; this keeps the helper faithful to
    Steno's parser, which preserves the body. Metadata values may be
    non-strings (lists/ints/bools) — coerce defensively downstream.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    body = text[m.end():]
    return meta, body


# ---------------------------------------------------------------------------
# Embedding model (cached)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Load (and cache) a SentenceTransformer by model name.

    Caching avoids reloading model weights + tokenizer on every call within a
    process. Imported lazily so memcore stays importable without the dep.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Cosine / distance math
# ---------------------------------------------------------------------------
def sim_from_distance(distance: float) -> float:
    """Recover cosine similarity from a ChromaDB cosine distance.

    Chroma's 'cosine' space stores distance = 1 - cosine_similarity, so the
    similarity is simply 1 - distance.
    """
    return 1.0 - distance


def cosine_matrix(embeddings):
    """Build a dense NxN cosine-similarity matrix from embedding vectors."""
    import numpy as np
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normalized = emb / (norms + 1e-8)
    return normalized @ normalized.T


# ---------------------------------------------------------------------------
# ChromaDB client / collection
# ---------------------------------------------------------------------------
def get_collection(store_dir, collection_name: str = DEFAULT_COLLECTION_NAME):
    """Get or create the shared memory collection (hnsw cosine).

    Creates the store directory if needed and returns a get_or_create
    collection so callers do not have to special-case first-run.
    """
    import chromadb
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={'hnsw:space': 'cosine'},
    )


# ---------------------------------------------------------------------------
# Store / collection resolution
# ---------------------------------------------------------------------------
def resolve_store(explicit, *env_vars, default=None):
    """Resolve a ChromaDB store dir.

    Precedence: explicit arg > each env var in order (e.g. STENO_STORE then
    MEMORY_STORE) > default. Returns a Path, or None if nothing resolved and no
    default given (callers may apply their own per-repo default, e.g. a
    memory-dir-relative .vigil/).
    """
    if explicit:
        return Path(explicit)
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return Path(val)
    if default is not None:
        return Path(default)
    return None


def resolve_collection(explicit: str | None = None) -> str:
    """Resolve the ChromaDB collection name.

    Precedence: explicit arg > MEMORY_COLLECTION env var > shared default.
    """
    if explicit:
        return explicit
    return os.environ.get('MEMORY_COLLECTION', DEFAULT_COLLECTION_NAME)
