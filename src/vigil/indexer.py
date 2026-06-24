"""Index markdown memory files into ChromaDB for semantic search.

Works with any directory of markdown files. Files with YAML frontmatter
(delimited by ---) get their metadata extracted; plain markdown works too.

Embeddings: all-MiniLM-L6-v2 (384-dim, CPU, ~80MB).
Storage: ChromaDB persistent client, one collection per memory directory.
Incremental: tracks file mtimes, only re-indexes changed files.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import yaml

from . import memcore

EMBED_MODEL = 'all-MiniLM-L6-v2'

# Shared default collection name. Steno indexes into the same default so Vigil
# can audit exactly what Steno builds. Override via the MEMORY_COLLECTION env
# var or the --collection CLI flag.
DEFAULT_COLLECTION_NAME = memcore.DEFAULT_COLLECTION_NAME
BATCH_SIZE = 64
MTIME_FILE = 'file_mtimes.json'

# Frontmatter pattern: leading --- ... --- block (matches Steno's parser).
FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)


def resolve_collection_name(collection_name: str | None = None) -> str:
    """Resolve the ChromaDB collection name.

    Precedence: explicit arg > MEMORY_COLLECTION env var > shared default.
    Delegates to the shared core.
    """
    return memcore.resolve_collection(collection_name)


# Backwards-compatible module-level default (resolves env at import time).
COLLECTION_NAME = resolve_collection_name()


def default_store_dir(memory_dir: Path) -> Path:
    """Default ChromaDB store location.

    Honors the MEMORY_STORE env var if set, otherwise .vigil/ inside the
    memory directory.
    """
    return memcore.resolve_store(
        None, 'MEMORY_STORE', default=memory_dir / '.vigil')


def _coerce_str(val) -> str:
    """Coerce a frontmatter value to a string safely.

    YAML values may be lists, ints, bools, None, etc. ChromaDB metadata and
    the provenance checks expect strings, so coerce defensively.
    """
    if val is None:
        return ''
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return ', '.join(_coerce_str(v) for v in val)
    if isinstance(val, bool):
        return 'true' if val else 'false'
    return str(val)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Uses pyyaml (yaml.safe_load) so nested YAML, lists, and multiline values
    parse correctly. Returns (metadata_dict, body_text). On missing or invalid
    frontmatter, returns ({}, full_text).

    NOTE: metadata values may be non-strings (lists, ints, bools). Consumers
    must guard/coerce — see _coerce_str.
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
    body = text[m.end():].strip()
    return meta, body


def chunk_text(text: str, source_file: str, meta: dict,
               max_chunk: int = 500) -> list[dict]:
    """Split text into chunks for embedding.

    Each chunk gets metadata for filtering. Chunks split on double-newlines
    (paragraph boundaries) and are capped at max_chunk characters.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ''

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chunk and current:
            chunks.append(current)
            current = para
        else:
            current = f'{current}\n\n{para}'.strip() if current else para

    if current:
        chunks.append(current)

    # Skip chunks that are too short to be meaningful
    MIN_CHUNK = 30
    results = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < MIN_CHUNK:
            continue
        # `type` may be top-level or nested under `metadata:` (Claude Code format).
        mtype = _coerce_str(meta.get('type', ''))
        if not mtype and isinstance(meta.get('metadata'), dict):
            mtype = _coerce_str(meta['metadata'].get('type', ''))
        results.append({
            'id': f'{source_file}:{i}',
            'text': chunk,
            'metadata': {
                'source_file': source_file,
                'chunk_index': i,
                'record_type': mtype,
                'memory_type': mtype,
                'name': _coerce_str(meta.get('name', '')),
                'access_count': 0,
                'last_accessed': '',
                'health_score': 1.0,
            }
        })

    return results


def _load_mtimes(store_dir: Path) -> dict:
    """Load file modification times from sidecar JSON."""
    path = store_dir / MTIME_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_mtimes(store_dir: Path, mtimes: dict):
    """Save file modification times to sidecar JSON."""
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / MTIME_FILE).write_text(json.dumps(mtimes, indent=2))


def build_index(
    memory_dir: Path,
    store_dir: Path | None = None,
    model_name: str = EMBED_MODEL,
    full_rebuild: bool = False,
    collection_name: str | None = None,
) -> dict:
    """Build or update ChromaDB index from a memory directory.

    Args:
        memory_dir: directory containing markdown files
        store_dir: ChromaDB storage location (default: memory_dir/.vigil/)
        model_name: sentence-transformer model name
        full_rebuild: if True, delete and rebuild entire index
        collection_name: ChromaDB collection (default: env/shared default)

    Returns:
        dict with stats: indexed, skipped, total, elapsed
    """
    if store_dir is None:
        store_dir = default_store_dir(memory_dir)
    collection_name = resolve_collection_name(collection_name)

    import chromadb
    from sentence_transformers import SentenceTransformer

    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))

    if full_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        collection_name,
        metadata={'hnsw:space': 'cosine'},
    )

    model = SentenceTransformer(model_name)
    mtimes = {} if full_rebuild else _load_mtimes(store_dir)

    t0 = time.time()
    indexed = 0
    skipped = 0

    md_files = sorted(memory_dir.glob('*.md'))

    for f in md_files:
        mtime = f.stat().st_mtime
        if f.name in mtimes and mtimes[f.name] >= mtime:
            skipped += 1
            continue

        content = f.read_text(encoding='utf-8', errors='replace')
        meta, body = parse_frontmatter(content)
        chunks = chunk_text(body, f.stem, meta)

        if not chunks:
            mtimes[f.name] = mtime
            continue

        # Remove old records for this file
        try:
            existing = collection.get(
                where={'source_file': f.stem},
                include=[],
            )
            if existing['ids']:
                collection.delete(ids=existing['ids'])
        except Exception:
            pass

        # Embed and add
        texts = [c['text'] for c in chunks]
        ids = [c['id'] for c in chunks]
        metas = [c['metadata'] for c in chunks]

        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            batch_ids = ids[start:start + BATCH_SIZE]
            batch_metas = metas[start:start + BATCH_SIZE]

            embeddings = model.encode(
                batch_texts, show_progress_bar=False
            ).tolist()

            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
            )

        mtimes[f.name] = mtime
        indexed += 1

    _save_mtimes(store_dir, mtimes)

    return {
        'indexed': indexed,
        'skipped': skipped,
        'total': len(md_files),
        'records': collection.count(),
        'elapsed': round(time.time() - t0, 1),
    }
