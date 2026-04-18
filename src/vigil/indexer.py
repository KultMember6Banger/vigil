"""Index markdown memory files into ChromaDB for semantic search.

Works with any directory of markdown files. Files with YAML frontmatter
(delimited by ---) get their metadata extracted; plain markdown works too.

Embeddings: all-MiniLM-L6-v2 (384-dim, CPU, ~80MB).
Storage: ChromaDB persistent client, one collection per memory directory.
Incremental: tracks file mtimes, only re-indexes changed files.
"""

import json
import re
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'vigil_memory'
BATCH_SIZE = 64
MTIME_FILE = 'file_mtimes.json'


def default_store_dir(memory_dir: Path) -> Path:
    """Default ChromaDB store location: .vigil/ inside the memory directory."""
    return memory_dir / '.vigil'


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Returns (metadata_dict, body_text). If no frontmatter found,
    returns ({}, full_text).
    """
    if not text.startswith('---'):
        return {}, text

    # Find closing ---
    end = text.find('---', 3)
    if end == -1:
        return {}, text

    front = text[3:end].strip()
    body = text[end + 3:].strip()

    meta = {}
    for line in front.split('\n'):
        line = line.strip()
        if ':' in line:
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip().strip('"').strip("'")

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
        results.append({
            'id': f'{source_file}:{i}',
            'text': chunk,
            'metadata': {
                'source_file': source_file,
                'chunk_index': i,
                'record_type': meta.get('type', ''),
                'memory_type': meta.get('type', ''),
                'name': meta.get('name', ''),
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
) -> dict:
    """Build or update ChromaDB index from a memory directory.

    Args:
        memory_dir: directory containing markdown files
        store_dir: ChromaDB storage location (default: memory_dir/.vigil/)
        model_name: sentence-transformer model name
        full_rebuild: if True, delete and rebuild entire index

    Returns:
        dict with stats: indexed, skipped, total, elapsed
    """
    if store_dir is None:
        store_dir = default_store_dir(memory_dir)

    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))

    if full_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
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
