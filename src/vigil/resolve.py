"""Resolution helpers — pick which file of a pair to keep / archive.

Shared by `vigil fix` (duplicates) and contradiction `suggested_keep`. The
ranking is deterministic and advisory: higher health_score wins; ties broken by
higher access_count, then more-recent mtime. Vigil NEVER deletes — the strongest
action is moving a file into an `archive/` subdir, and only under `--apply`.
"""

from __future__ import annotations

from pathlib import Path


def file_signals(memory_dir: Path, stem: str,
                 access_data: dict | None = None,
                 health_data: dict | None = None) -> dict:
    """Gather ranking signals for a memory file stem.

    Returns {stem, mtime, access_count, health_score, exists}. Missing files
    yield mtime 0 / exists False so they always lose to a present file.
    """
    access_data = access_data or {}
    health_data = health_data or {}
    path = Path(memory_dir) / f'{stem}.md'
    exists = path.exists()
    mtime = path.stat().st_mtime if exists else 0.0
    acc = access_data.get(stem, {})
    return {
        'stem': stem,
        'mtime': mtime,
        'access_count': int(acc.get('access_count', 0)),
        'health_score': float(health_data.get(stem, 1.0)),
        'exists': exists,
    }


def _rank_key(sig: dict) -> tuple:
    # Higher health, then higher access, then newer mtime, then existence.
    return (sig['health_score'], sig['access_count'], sig['mtime'],
            1 if sig['exists'] else 0)


def choose_keep(sig_a: dict, sig_b: dict) -> tuple[str, str]:
    """Return (keep_stem, archive_stem) for a pair of signal dicts.

    Higher health_score wins; ties broken by access_count then mtime. Stable:
    if fully tied, keeps the first argument.
    """
    if _rank_key(sig_b) > _rank_key(sig_a):
        return sig_b['stem'], sig_a['stem']
    return sig_a['stem'], sig_b['stem']


def load_health_data(store_dir: Path, collection_name: str | None = None) -> dict:
    """Pull per-file health_score from ChromaDB metadata (max across chunks).

    Best-effort: returns {} if the store/collection is unavailable.
    """
    try:
        import chromadb
        from .indexer import resolve_collection_name
        collection_name = resolve_collection_name(collection_name)
        client = chromadb.PersistentClient(path=str(store_dir))
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return {}
        recs = collection.get(include=['metadatas'], limit=count)
    except Exception:
        return {}

    health: dict = {}
    for meta in recs['metadatas']:
        src = meta.get('source_file', '')
        if not src:
            continue
        hs = float(meta.get('health_score', 1.0))
        if src not in health or hs > health[src]:
            health[src] = hs
    return health
