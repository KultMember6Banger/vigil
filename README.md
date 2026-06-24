# Vigil

Memory health monitor for AI agents. Detects contradictions, duplicates, staleness, and orphan references in markdown-based memory stores.

**v0.3.0** — Vigil now *acts*, not just detects: `vigil fix` turns a scan into a
resolution plan and (with `--apply`) archives stale/duplicate files (never
deletes); a `vigil serve` daemon + git pre-commit `hook` give sub-second
real-time contradiction gating; `vigil history` tracks health trends over time;
and an optional `.vigil.toml` tunes every threshold. Builds on v0.2.0's
configurable store/collection (`--store` / `--collection`, `MEMORY_STORE` /
`MEMORY_COLLECTION`), Steno-shared `agent_memory` default, MCP server, PyYAML
frontmatter parsing, and corrected cosine/staleness math.

## The problem

AI agents with persistent memory accumulate drift over time. Memories contradict each other. Stale facts linger. References break. Duplicates waste context. No existing tool audits memory health after the fact — Vigil is the first.

## What it does

Point Vigil at a directory of markdown files (with optional YAML frontmatter) and it finds:

- **Contradictions** — memory pairs asserting conflicting facts, detected via NLI cross-encoder (DeBERTa)
- **Duplicates** — near-identical memories across different files (cosine similarity > 0.85)
- **Isolated entries** — memories with no semantic neighbors, effectively unreachable by retrieval
- **Stale memories** — Ebbinghaus-informed exponential decay, factoring in access frequency
- **Orphan references** — broken file paths and cross-references to memories that don't exist
- **Missing provenance** — files without required metadata (name, type, description)

Plus:

- **Pre-write check** — real-time gate that catches contradictions *before* you write a new memory
- **Health scores** — per-file health score (0.1 to 1.0) written to ChromaDB metadata, usable by downstream RAG to deprioritize unhealthy memories
- **Access tracking** — records when and how often each memory is retrieved, feeds staleness scoring

And, new in v0.3.0 — Vigil *resolves*, not just reports:

- **`vigil fix`** — turns a scan into a RESOLUTION PLAN (dry-run by default).
  With `--apply` it **archives** stale/high-confidence-duplicate files (moves
  them into an `archive/` subdir — **never deletes**); contradictions, orphans,
  and provenance gaps are surfaced as advisory actions only.
- **`vigil serve` + `vigil hook`** — a long-lived daemon loads the models once
  and answers pre-write checks over a unix socket; a git pre-commit hook blocks
  commits that introduce **CRITICAL** contradictions. Sub-second checks instead
  of a multi-second model reload every call.
- **`vigil history`** — appends a health snapshot on every `vigil health` run and
  shows the trend over time: overall health plus which files improved/degraded.
- **`.vigil.toml`** — optional per-directory config for every threshold,
  staleness weight, volatility marker, required provenance field, and ignore glob.

## Quickstart

```bash
pip install vigil-memory

# 1. Index your memory directory
vigil index ./memory/

# 2. Run a health scan
vigil scan ./memory/

# 3. Check new text before writing
vigil check ./memory/ "The database uses PostgreSQL for auth"

# 4. Full scan + update health scores (also records a history snapshot)
vigil health ./memory/

# 5. Build a resolution plan (dry-run), then apply it (archives only)
vigil fix ./memory/
vigil fix ./memory/ --apply --yes

# 6. See how health is trending over time
vigil history ./memory/

# 7. Run the always-on pre-write daemon, then check against it
vigil serve ./memory/ &                       # loads models once
vigil check ./memory/ "New fact" --daemon     # sub-second check

# 8. Install the git pre-commit gate (blocks committing CRITICAL contradictions)
vigil hook install ./memory/
```

## How it works

### Contradiction detection

1. All memory chunks are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB
2. Pairwise cosine similarity identifies same-topic pairs across different files (0.65-0.90 range)
3. Core factual assertions are extracted from each chunk (skipping headers, lists, cross-refs)
4. NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`) classifies pairs as contradiction/entailment/neutral
5. Entity overlap filter requires shared specific terms to reduce false positives
6. Results capped at top 25 most confident contradictions

### Staleness scoring (Ebbinghaus-informed)

Uses exponential decay (not linear) with three signals:
- **Effective age** — days since last access or modification, whichever is newer
- **Content age** — newest date found in the text
- **Volatility** — presence of temporal markers (`current`, `pending`, `status:`, `TODO`)

Retention follows the Ebbinghaus forgetting curve: `retention = e^(-t/s)` where `s` (strength) increases with each access (spaced repetition effect). A frequently-accessed 90-day-old memory stays healthier than a never-accessed 14-day-old one.

When ChromaDB is available, access tracking data enriches the score. Without it, falls back to file-age-only mode.

### Health-weighted RAG

Vigil writes per-file health scores into ChromaDB metadata. Your RAG retrieval can use these to deprioritize unhealthy memories:

```python
# In your retrieval code (ChromaDB cosine distance = 1 - cosine_similarity):
raw_score = 1 - distance  # cosine similarity
health = float(metadata.get('health_score', 1.0))
final_score = raw_score * health  # unhealthy memories rank lower
```

[Steno](https://github.com/KultMember6Banger/steno) has this built in — its retrieval engine applies health-weighted scoring automatically.

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

Frontmatter is parsed with PyYAML, so nested values, lists, and multiline
strings are supported. Non-string metadata values are coerced where checks
expect strings.

## Store & collection configuration

Every command resolves its ChromaDB **store directory** and **collection name**
the same way:

| Setting | `--flag` | Env var | Default |
|---------|----------|---------|---------|
| Store dir | `--store <path>` | `MEMORY_STORE` | `<memory_dir>/.vigil/` |
| Collection | `--collection <name>` | `MEMORY_COLLECTION` | `agent_memory` |

Precedence is flag > env var > default. The shared default collection name
(`agent_memory`) lets Vigil audit the same store another tool indexes into.

## Steno integration

[Steno](https://github.com/KultMember6Banger/steno) indexes memory into the
same ChromaDB. Point both at one store + collection and Vigil scores exactly
what Steno built:

```bash
# Steno builds the index...
steno index ./memory/ --store ./shared --collection agent_memory

# ...Vigil audits and scores the very same records:
vigil health ./memory/ --store ./shared --collection agent_memory
```

Or set it once via env vars so every command agrees:

```bash
export MEMORY_STORE=./shared
export MEMORY_COLLECTION=agent_memory
vigil scan ./memory/
```

Vigil reads `access_count` / `last_accessed` from each record's metadata and
writes back `health_score`. Steno's retrieval applies `final_score =
similarity * health_score` automatically.

## MCP server

Vigil ships a stdio JSON-RPC **MCP server** (protocol `2024-11-05`) exposing
its checks as native AI-agent tools, installed as the `vigil-mcp` console
script (module: `mcp_server.py`).

Tools:

| Tool | Purpose |
|------|---------|
| `vigil_scan` | Run a full or selective health scan → issues as JSON |
| `vigil_check` | Pre-write contradiction gate for proposed new text → conflicts |
| `vigil_health` | Full scan + write updated health scores into ChromaDB |

Each tool accepts `memory_dir` (and optional `store` / `collection`) and
returns `{"content": [{"type": "text", "text": <json>}]}`.

Example MCP client config (Claude Desktop / Claude Code):

```json
{
  "mcpServers": {
    "vigil": {
      "command": "vigil-mcp",
      "env": { "MEMORY_STORE": "./shared", "MEMORY_COLLECTION": "agent_memory" }
    }
  }
}
```

Without the console script, run it directly: `python mcp_server.py` (ensure the
`vigil` package is importable, e.g. `PYTHONPATH=src`).

## CLI reference

### `vigil index <memory_dir>`

Build or update the ChromaDB search index.

| Flag | Description |
|------|-------------|
| `--store <path>` | Custom ChromaDB location (default: `$MEMORY_STORE` or `<memory_dir>/.vigil/`) |
| `--collection <name>` | Collection name (default: `$MEMORY_COLLECTION` or `agent_memory`) |
| `--rebuild` | Delete and rebuild entire index |

### `vigil scan <memory_dir>`

Run health checks.

| Flag | Description |
|------|-------------|
| `--check <type>` | Run specific check(s): `contradictions`, `duplicates`, `isolated`, `stale`, `orphans`, `provenance` |
| `--json` | Output as JSON |
| `--store <path>` | Custom ChromaDB location (default: `$MEMORY_STORE` or `<memory_dir>/.vigil/`) |
| `--collection <name>` | Collection name (default: `$MEMORY_COLLECTION` or `agent_memory`) |

### `vigil check <memory_dir> "text"`

Pre-write contradiction check.

| Flag | Description |
|------|-------------|
| `--file <path>` | Read text from file instead of argument |
| `--source <stem>` | Exclude this file from comparison |
| `--store <path>` | Custom ChromaDB location (default: `$MEMORY_STORE` or `<memory_dir>/.vigil/`) |
| `--collection <name>` | Collection name (default: `$MEMORY_COLLECTION` or `agent_memory`) |

### `vigil health <memory_dir>`

Full scan + write health scores to ChromaDB, **and append a history snapshot**
to `<store>/history.jsonl`. Honors `--store` / `--collection` / `--config` (and
`MEMORY_STORE` / `MEMORY_COLLECTION`) like the other commands.

### `vigil fix <memory_dir>`

Run a scan and emit a **resolution plan**. Dry-run by default; `--apply` performs
the strongest action Vigil ever takes — **moving** a file into an `archive/`
subdir (it is *moved*, never deleted, so it is fully reversible).

| Flag | Description |
|------|-------------|
| `--apply` | Apply the plan (archive only — never deletes) |
| `--yes` | Skip the confirmation prompt when applying |
| `--json` | Output the plan as JSON |
| `--store` / `--collection` / `--config` | As for other commands |

Per category:

| Category | Action |
|----------|--------|
| **stale** | `--apply` archives the file. Dry-run lists what *would* be archived. |
| **duplicates** | Keeps the higher-health (else more-recently-modified) file of the pair, archives the loser — **only** when `sim >= fix.duplicate_threshold` (0.92 by default). Below that, listed for review. |
| **orphans** | Lists each broken ref with `file.md:line` and suggests manual removal. Vigil never auto-edits file bodies. |
| **contradictions** | Surfaces `suggested_keep` (newer mtime + higher access_count + higher health) — **advisory only**, never auto-archived. |
| **provenance** | Lists the missing fields per file (it can't infer the values). |

### `vigil history <memory_dir>`

Show the health trend across recorded snapshots: overall health over time plus
which files **improved / degraded** since the previous run. Dependency-free
(reads `<store>/history.jsonl`). `--json` dumps the raw snapshots.

### `vigil serve <memory_dir>`

Run a long-lived **pre-write check daemon**. It loads the embedder + NLI model
*once* and answers checks over a local unix-domain socket using newline-delimited
JSON, so the hook / your agent gets sub-second checks instead of paying the
model-load cost every call.

| Flag | Description |
|------|-------------|
| `--socket <path>` | Socket path (default: `<store>/vigil.sock`, auto-relocated to a short temp path if the in-store path is too long for AF_UNIX) |
| `--store` / `--collection` | As for other commands |

Protocol (one JSON object per line, both ways):

```
request : {"text": "...", "source": "<stem to exclude>", "collection": null}
response: {"ok": true, "conflict_count": N, "conflicts": [ ...issue dicts... ]}
```

Clients connect via `vigil check ... --daemon` (which falls back to an in-process
check if no daemon is running), or directly via `vigil.daemon.check_via_daemon`.

### `vigil hook install|run <memory_dir>`

`vigil hook install` writes a git **pre-commit** hook. If the memory dir is in a
git repo, the hook lands in that repo's `.git/hooks/pre-commit` (chaining onto an
existing hook rather than clobbering it); otherwise a standalone script is
written to `<memory_dir>/.vigil/pre-commit` for you to wire up.

The hook runs `vigil hook run`, which pre-write-checks the **staged** (or, outside
git, all) `*.md` files and:

- **blocks the commit** (nonzero exit) on any **CRITICAL** contradiction,
- **warns** but allows the commit on **WARNING**-level conflicts.

If a `vigil serve` daemon is running it is used automatically for speed; else the
hook falls back to an in-process check. Override a block with
`git commit --no-verify`.

## `.vigil.toml` config

Drop an optional `.vigil.toml` in your memory directory (or pass `--config
<path>`) to tune any threshold. **Precedence: CLI flags > `.vigil.toml` >
built-in defaults.** A missing file is never an error — you only override what
you set. A fully-commented template ships as
[`.vigil.toml.example`](.vigil.toml.example).

```toml
# every value shown with its default
required_provenance_fields = ["name", "type", "description"]
ignore_globs = []                       # files skipped in ALL checks (globs)

[contradictions]
sim_low = 0.65                          # same-topic similarity window
sim_high = 0.90
nli_threshold = 0.85                    # min NLI contradiction probability

[isolated]
isolation_threshold = 0.3               # below this best-cross-file-sim = isolated

[staleness]
warn_days = 14
critical_days = 30
base_weight = 0.4                       # blend weights for the staleness score
content_weight = 0.3
volatility_weight = 0.3
[[staleness.volatility_markers]]        # repeatable; replaces the default list
pattern = "\\b(current|currently|ongoing|active)\\b"
label = "temporal_state"
weight = 0.3

[duplicates]
threshold = 0.85                        # flag near-duplicates at/above this

[fix]
archive_dir = "archive"                 # where `vigil fix --apply` moves files
duplicate_threshold = 0.92              # only auto-archive dupes at/above this
```

| Key | Default | Affects |
|-----|---------|---------|
| `required_provenance_fields` | `["name","type","description"]` | provenance |
| `ignore_globs` | `[]` | all checks |
| `contradictions.sim_low` / `sim_high` / `nli_threshold` | 0.65 / 0.90 / 0.85 | contradictions |
| `isolated.isolation_threshold` | 0.3 | isolated |
| `staleness.warn_days` / `critical_days` | 14 / 30 | stale |
| `staleness.base_weight` / `content_weight` / `volatility_weight` | 0.4 / 0.3 / 0.3 | stale |
| `staleness.volatility_markers` | 4 built-in markers | stale |
| `duplicates.threshold` | 0.85 | duplicates |
| `fix.archive_dir` | `"archive"` | fix |
| `fix.duplicate_threshold` | 0.92 | fix |

The config is parsed with `tomllib` (Python 3.11+) or `tomli` if installed;
Vigil also ships a tiny dependency-free fallback parser for the subset it needs,
so the config works even where neither is available.

## Python API

```python
from vigil.scanner import (
    find_contradictions, find_stale, find_orphans,
    find_duplicates, find_isolated, find_unprovenanced,
    pre_write_check,
)
from vigil.indexer import build_index

# Index
build_index(Path('./memory/'))

store = Path('./memory/.vigil/')

# Lightweight checks (no heavy deps)
stale = find_stale(Path('./memory/'))
orphans = find_orphans(Path('./memory/'))
unprovenanced = find_unprovenanced(Path('./memory/'))

# Enriched staleness (with access frequency from ChromaDB)
stale_enriched = find_stale(Path('./memory/'), store_dir=store)

# Heavy checks (require ChromaDB + embeddings)
duplicates = find_duplicates(store)
isolated = find_isolated(store)

# Pre-write gate
issues = pre_write_check(
    "The API uses Firebase for auth",
    store_dir=store,
)
for issue in issues:
    print(f"[{issue.severity}] {issue.message}")
```

`find_stale()`, `find_orphans()`, and `find_unprovenanced()` have no heavy dependencies — they work with just the standard library. Contradiction/duplicate/isolated detection and pre-write checks require `chromadb` and `sentence-transformers`.

The v0.3.0 modules are equally importable:

```python
from vigil.config import VigilConfig
from vigil.history import record_health_snapshot, load_history, format_history
from vigil.fix import run_fix, build_plan, apply_plan, format_plan
from vigil import daemon  # daemon.serve / daemon.check_via_daemon / daemon.daemon_running

cfg = VigilConfig.load(Path('./memory/'))          # reads .vigil.toml if present
cfg.override(sim_high=0.92)                          # layer a flag on top

# A scan with config + a fixed `now` (deterministic snapshots/tests):
from vigil.scanner import full_scan
results = full_scan(Path('./memory/'), config=cfg)

# Plan + apply (archives stale/duplicate files; never deletes):
actions, _ = run_fix(Path('./memory/'), store, cfg, apply=True)
```

`history.record_health_snapshot(...)` takes the timestamp as an argument (never
`datetime.now()` at import time), so callers control it and tests stay
deterministic.

## Requirements

- Python 3.10+
- `chromadb >= 0.4.0`
- `sentence-transformers >= 2.2.0`
- `pyyaml >= 6.0`
- `tomli >= 2.0.0` on Python < 3.11 (for `.vigil.toml`; a built-in fallback
  parser covers the case where it is absent)

The lightweight checks (stale / orphans / provenance), the config loader, the
history trend, the `fix` planner/archiver, and the pre-commit `hook` gate need
only `pyyaml` (plus the stdlib); the vector checks and the `serve` daemon add
`chromadb` + `sentence-transformers`.

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
