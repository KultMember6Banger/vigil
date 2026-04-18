# Vigil

Memory health monitor for AI agents. Detects contradictions, duplicates, staleness, and orphan references in markdown-based memory stores.

## The problem

AI agents with persistent memory accumulate drift over time. Memories contradict each other. Stale facts linger. References break. Duplicates waste context. No existing tool audits memory health after the fact — Vigil is the first.

## What it does

Point Vigil at a directory of markdown files (with optional YAML frontmatter) and it finds:

- **Contradictions** — memory pairs asserting conflicting facts, detected via NLI cross-encoder (DeBERTa)
- **Duplicates** — near-identical memories across different files (cosine similarity > 0.85)
- **Stale memories** — likely outdated based on file age, content dates, and volatility markers
- **Orphan references** — broken file paths and cross-references to memories that don't exist

Plus:

- **Pre-write check** — real-time gate that catches contradictions *before* you write a new memory
- **Health scores** — per-file health score (0.1 to 1.0) written to ChromaDB metadata, usable by downstream RAG to deprioritize unhealthy memories

## Quickstart

```bash
pip install vigil-memory

# 1. Index your memory directory
vigil index ./memory/

# 2. Run a health scan
vigil scan ./memory/

# 3. Check new text before writing
vigil check ./memory/ "The database uses PostgreSQL for auth"

# 4. Full scan + update health scores
vigil health ./memory/
```

## How it works

### Contradiction detection

1. All memory chunks are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB
2. Pairwise cosine similarity identifies same-topic pairs across different files (0.65-0.90 range)
3. Core factual assertions are extracted from each chunk (skipping headers, lists, cross-refs)
4. NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`) classifies pairs as contradiction/entailment/neutral
5. Entity overlap filter requires shared specific terms to reduce false positives
6. Results capped at top 25 most confident contradictions

### Staleness scoring

Composite score from three signals:
- **File age** — days since last modification
- **Content age** — newest date found in the text
- **Volatility** — presence of temporal markers (`current`, `pending`, `status:`, `TODO`)

Formula: `staleness = (file_age/30) * 0.4 + (content_age/30) * 0.3 + min(volatility, 1.0) * 0.3`

### Health-weighted RAG

Vigil writes per-file health scores into ChromaDB metadata. Your RAG retrieval can use these to deprioritize unhealthy memories:

```python
# In your retrieval code:
raw_score = 1 - (distance / 2)  # cosine similarity
health = float(metadata.get('health_score', 1.0))
final_score = raw_score * health  # unhealthy memories rank lower
```

## Memory file format

Vigil works with any markdown files. Files with YAML frontmatter get richer metadata:

```markdown
---
name: My memory title
type: feedback
description: One-line description
---

The actual memory content goes here.
```

If no frontmatter is present, the entire file is treated as content.

## CLI reference

### `vigil index <memory_dir>`

Build or update the ChromaDB search index.

| Flag | Description |
|------|-------------|
| `--store <path>` | Custom ChromaDB location (default: `<memory_dir>/.vigil/`) |
| `--rebuild` | Delete and rebuild entire index |

### `vigil scan <memory_dir>`

Run health checks.

| Flag | Description |
|------|-------------|
| `--check <type>` | Run specific check(s): `contradictions`, `duplicates`, `stale`, `orphans` |
| `--json` | Output as JSON |
| `--store <path>` | Custom ChromaDB location |

### `vigil check <memory_dir> "text"`

Pre-write contradiction check.

| Flag | Description |
|------|-------------|
| `--file <path>` | Read text from file instead of argument |
| `--source <stem>` | Exclude this file from comparison |
| `--store <path>` | Custom ChromaDB location |

### `vigil health <memory_dir>`

Full scan + write health scores to ChromaDB.

## Python API

```python
from vigil.scanner import find_contradictions, find_stale, find_orphans, pre_write_check
from vigil.indexer import build_index

# Index
build_index(Path('./memory/'))

# Scan
stale = find_stale(Path('./memory/'))
orphans = find_orphans(Path('./memory/'))

# Pre-write gate
issues = pre_write_check(
    "The API uses Firebase for auth",
    store_dir=Path('./memory/.vigil/'),
)
for issue in issues:
    print(f"[{issue.severity}] {issue.message}")
```

`find_stale()` and `find_orphans()` have no heavy dependencies — they work with just the standard library. Contradiction detection and pre-write checks require `chromadb` and `sentence-transformers`.

## Requirements

- Python 3.10+
- `chromadb >= 0.4.0`
- `sentence-transformers >= 2.2.0`

Models are downloaded automatically on first use:
- `all-MiniLM-L6-v2` (~80MB) — embeddings
- `cross-encoder/nli-deberta-v3-xsmall` (~90MB) — contradiction detection

## Prior art

No dedicated memory health monitor for AI agents existed before Vigil (confirmed via research, April 2026). Related but different:

- **Zep/Graphiti** — inline contradiction on write, not post-hoc audit
- **mem0** — ADD-only, health features planned not implemented
- **SimpleMem** — dedup via recursive consolidation, no standalone health
- **doobidoo/mcp-memory-service** — "contradicts" edge type exists, no audit CLI

## License

Apache 2.0
