"""Core health checks — contradictions, duplicates, staleness, orphans.

Each check returns a list of Issue objects. All checks are independent
and can be run individually or together via full_scan().

Heavy dependencies (chromadb, numpy, sentence-transformers) are imported
inside functions that need them. find_stale() and find_orphans() work
with only the standard library + the indexer's frontmatter parser.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .indexer import (
    parse_frontmatter, COLLECTION_NAME, EMBED_MODEL,
    default_store_dir,
)

# NLI model for contradiction detection
NLI_MODEL = 'cross-encoder/nli-deberta-v3-xsmall'


@dataclass
class Issue:
    """A single health issue found by Vigil."""
    severity: str       # CRITICAL, WARNING, INFO
    category: str       # contradiction, duplicate, stale, orphan, pre_write_conflict
    message: str
    files: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


# --- Shared utilities ---

def _get_all_records(store_dir: Path):
    """Pull all records from ChromaDB with embeddings."""
    import chromadb
    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    if count == 0:
        return None
    return collection.get(
        include=['documents', 'metadatas', 'embeddings'],
        limit=count,
    )


def _build_sim_matrix(embeddings):
    """Build cosine similarity matrix from embedding vectors."""
    import numpy as np
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normalized = emb / (norms + 1e-8)
    return normalized @ normalized.T


def _softmax(x):
    """Numerically stable softmax."""
    import numpy as np
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _core_assertion(text, max_len=150):
    """Extract the first factual assertion from a chunk.

    NLI cross-encoders work best on sentence-level input, not paragraphs.
    Skips headers, lists, cross-references, and structural lines.
    """
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith(('#', '|', '---', '- [', '- `', '> ', '* ')):
            continue
        if line.startswith('- ') and len(line) < 80:
            continue
        if re.match(r'^\d+\.', line):
            continue
        if any(x in line.lower() for x in ['cross-ref', 'see also', 'source:', 'see `']):
            continue
        if re.search(r'\b(is|are|was|were|has|have|does|do|use|run|should|must|can|never)\b', line, re.I):
            return line[:max_len]
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            return line[:max_len]
    return text[:max_len]


def _extract_entities(text):
    """Extract specific entities for overlap checking."""
    entities = set()
    entities.update(re.findall(r'[a-z_]+\.(?:md|py|sh|json|yml)', text))
    entities.update(
        w.lower() for w in re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        if w not in ('The', 'This', 'That', 'What', 'When', 'Where',
                     'How', 'Why', 'STOP', 'NOT', 'NEVER', 'HARD',
                     'RED', 'WARNING', 'CRITICAL', 'INFO',
                     'Rule', 'New', 'See', 'Use', 'Run', 'Check',
                     'Before', 'After')
    )
    entities.update(re.findall(r'port\s*\d+|\b\d{4,5}\b', text))
    return entities


def _line_context(text: str, pos: int, ctx: int = 50) -> str:
    """Extract surrounding text for a match position."""
    start = max(0, pos - ctx)
    end = min(len(text), pos + ctx)
    return text[start:end].replace('\n', ' ').strip()


# --- 1. Contradiction Detection ---

def find_contradictions(
    store_dir: Path,
    sim_low: float = 0.65,
    sim_high: float = 0.90,
    nli_threshold: float = 0.85,
) -> list[Issue]:
    """Find memory pairs that contradict each other.

    Uses pairwise cosine similarity to find same-topic pairs, then
    NLI cross-encoder to classify as contradiction/entailment/neutral.
    Entity overlap filter reduces false positives.
    """
    import numpy as np

    records = _get_all_records(store_dir)
    if not records or not records['ids']:
        return []

    try:
        from sentence_transformers import CrossEncoder
        nli = CrossEncoder(NLI_MODEL)
    except Exception as e:
        print(f'  WARN: NLI model unavailable ({e}), skipping contradictions')
        return []

    n = len(records['ids'])
    sim_matrix = _build_sim_matrix(records['embeddings'])

    candidates = []
    seen = set()

    for i in range(n):
        src_i = records['metadatas'][i].get('source_file', '')
        for j in range(i + 1, n):
            src_j = records['metadatas'][j].get('source_file', '')
            if src_i == src_j:
                continue
            sim = float(sim_matrix[i, j])
            if sim < sim_low or sim > sim_high:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            candidates.append((i, j, sim))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:500]

    sentence_pairs = []
    for i, j, _ in candidates:
        text_a = _core_assertion(records['documents'][i])
        text_b = _core_assertion(records['documents'][j])
        sentence_pairs.append((text_a, text_b))

    BATCH = 32
    all_logits = []
    for start in range(0, len(sentence_pairs), BATCH):
        batch = sentence_pairs[start:start + BATCH]
        logits = nli.predict(batch)
        all_logits.extend(logits)

    probs = _softmax(np.array(all_logits))

    issues = []
    for idx, (i, j, sim) in enumerate(candidates):
        c_prob = float(probs[idx, 0])
        if c_prob < nli_threshold:
            continue

        ent_a = _extract_entities(records['documents'][i])
        ent_b = _extract_entities(records['documents'][j])
        overlap = ent_a & ent_b
        if not overlap:
            continue

        severity = 'CRITICAL' if c_prob > 0.95 else 'WARNING'
        issues.append(Issue(
            severity=severity,
            category='contradiction',
            message=f'Contradiction (NLI={c_prob:.2f}, sim={sim:.2f})',
            files=[
                records['metadatas'][i].get('source_file', ''),
                records['metadatas'][j].get('source_file', ''),
            ],
            details={
                'text_a': records['documents'][i][:200],
                'text_b': records['documents'][j][:200],
                'nli_score': round(c_prob, 3),
                'similarity': round(sim, 3),
                'shared_entities': list(overlap)[:5],
            }
        ))

    issues.sort(key=lambda x: x.details.get('nli_score', 0), reverse=True)
    return issues[:25]


# --- 2. Duplicate Detection ---

def find_duplicates(
    store_dir: Path,
    threshold: float = 0.85,
) -> list[Issue]:
    """Find near-duplicate memories across different files."""
    records = _get_all_records(store_dir)
    if not records or not records['ids']:
        return []

    n = len(records['ids'])
    sim_matrix = _build_sim_matrix(records['embeddings'])

    issues = []
    seen = set()
    MIN_TEXT = 60

    for i in range(n):
        if len(records['documents'][i].strip()) < MIN_TEXT:
            continue
        src_i = records['metadatas'][i].get('source_file', '')
        for j in range(i + 1, n):
            if len(records['documents'][j].strip()) < MIN_TEXT:
                continue
            src_j = records['metadatas'][j].get('source_file', '')
            if src_i == src_j:
                continue
            sim = float(sim_matrix[i, j])
            if sim < threshold:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            issues.append(Issue(
                severity='WARNING',
                category='duplicate',
                message=f'Near-duplicate (sim={sim:.2f})',
                files=[src_i, src_j],
                details={
                    'text_a': records['documents'][i][:200],
                    'text_b': records['documents'][j][:200],
                    'similarity': round(sim, 3),
                }
            ))

    return sorted(issues, key=lambda x: x.details.get('similarity', 0), reverse=True)


# --- 2b. Isolated Entry Detection ---

def find_isolated(
    store_dir: Path,
    isolation_threshold: float = 0.3,
) -> list[Issue]:
    """Find memory entries with no semantic neighbors.

    Brain analog: hippocampus-dropped records — facts with no binding
    associations that will never surface via retrieval. An entry whose
    max similarity to any other entry is below the threshold has no
    retrieval path and is effectively dead.

    Requires ChromaDB + embeddings.
    """
    records = _get_all_records(store_dir)
    if not records or not records['ids']:
        return []

    import numpy as np

    n = len(records['ids'])
    if n < 2:
        return []

    sim_matrix = _build_sim_matrix(records['embeddings'])

    # Per-file: best similarity to any chunk from a DIFFERENT file
    file_best_sim = {}
    for i in range(n):
        src_i = records['metadatas'][i].get('source_file', '')
        if not src_i:
            continue
        for j in range(n):
            if i == j:
                continue
            src_j = records['metadatas'][j].get('source_file', '')
            if src_i == src_j:
                continue
            sim = float(sim_matrix[i, j])
            if src_i not in file_best_sim or sim > file_best_sim[src_i]:
                file_best_sim[src_i] = sim

    issues = []
    for src, best_sim in sorted(file_best_sim.items()):
        if best_sim < isolation_threshold:
            issues.append(Issue(
                severity='INFO',
                category='isolated',
                message=f'Isolated entry (max_sim={best_sim:.2f}, threshold={isolation_threshold})',
                files=[src],
                details={
                    'max_similarity': round(best_sim, 3),
                    'threshold': isolation_threshold,
                }
            ))

    return sorted(issues, key=lambda x: x.details.get('max_similarity', 0))


# --- 3. Staleness Scoring (Ebbinghaus-informed) ---

import math


def _ebbinghaus_retention(age_days: float, strength: float = 1.0) -> float:
    """Ebbinghaus forgetting curve: retention = e^(-t/s).

    age_days: time since last write or access
    strength: reinforcement factor (higher = slower decay).
              Each access multiplies strength by 1.5.
    Returns: 0.0 (completely forgotten) to 1.0 (perfectly fresh).
    """
    if age_days <= 0:
        return 1.0
    return math.exp(-age_days / max(strength * 30, 1))


def _get_access_data(store_dir: Path) -> dict[str, dict]:
    """Pull access_count and last_accessed from ChromaDB metadata.

    Returns {source_file: {access_count, last_accessed, ...}} aggregated
    across all chunks for each file (max access_count, latest last_accessed).
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(store_dir))
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            return {}
        records = collection.get(include=['metadatas'], limit=count)
    except Exception:
        return {}

    access = {}
    for meta in records['metadatas']:
        src = meta.get('source_file', '')
        if not src:
            continue
        ac = int(meta.get('access_count', 0))
        la = meta.get('last_accessed', '')
        if src not in access:
            access[src] = {'access_count': ac, 'last_accessed': la}
        else:
            access[src]['access_count'] = max(access[src]['access_count'], ac)
            if la > access[src]['last_accessed']:
                access[src]['last_accessed'] = la
    return access


def find_stale(
    memory_dir: Path,
    warn_days: int = 14,
    critical_days: int = 30,
    store_dir: Path | None = None,
) -> list[Issue]:
    """Score memories for staleness using Ebbinghaus-informed decay.

    Three signals:
    - File/content age with exponential decay (not linear)
    - Volatility markers (temporal words, TODOs, status fields)
    - Access frequency (if ChromaDB available via store_dir)

    Access resets the decay curve: a 60-day-old file accessed yesterday
    is fresher than a 20-day-old file never accessed.

    No heavy dependencies for base operation — ChromaDB only used
    when store_dir is provided for access-frequency enrichment.
    """
    issues = []
    now = datetime.now()

    # Pull access data if ChromaDB available
    access_data = {}
    if store_dir is not None:
        access_data = _get_access_data(store_dir)

    VOLATILE = [
        (r'\b(current|currently|ongoing|in.?progress|active)\b', 'temporal_state', 0.3),
        (r'\b(pending|waiting|queued|blocked|next)\b', 'pending_action', 0.2),
        (r'\bstatus:\s*\w+', 'explicit_status', 0.3),
        (r'\b(todo|TODO|FIXME|HACK|TEMP)\b', 'action_marker', 0.2),
    ]

    DATE_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')

    for f in sorted(memory_dir.glob('*.md')):
        if f.name in ('MEMORY.md', 'README.md'):
            continue

        content = f.read_text(encoding='utf-8', errors='replace')
        _, body = parse_frontmatter(content)

        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        age_days = (now - mtime).days

        if age_days < warn_days:
            continue

        # Volatility score
        volatility = 0.0
        markers = []
        for pattern, label, weight in VOLATILE:
            hits = re.findall(pattern, body, re.IGNORECASE)
            if hits:
                volatility += weight * len(hits)
                markers.append(f'{label}({len(hits)})')

        # Newest date in content
        dates = []
        for m in DATE_RE.finditer(body):
            try:
                dates.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                continue
        newest = max(dates) if dates else mtime
        content_age = (now - newest).days

        # Access-frequency enrichment
        file_access = access_data.get(f.stem, {})
        access_count = file_access.get('access_count', 0)
        last_accessed_str = file_access.get('last_accessed', '')

        # Effective age: time since last access OR last write, whichever is newer
        effective_age = age_days
        if last_accessed_str:
            try:
                la_dt = datetime.fromisoformat(last_accessed_str.replace('Z', '+00:00'))
                la_naive = la_dt.replace(tzinfo=None)
                access_age = (now - la_naive).days
                effective_age = min(effective_age, access_age)
            except (ValueError, TypeError):
                pass

        # Ebbinghaus: access count boosts retention strength
        # Each access multiplies strength by 1.5 (spaced repetition effect)
        strength = 1.5 ** min(access_count, 10)  # cap at 10 to prevent overflow
        retention = _ebbinghaus_retention(effective_age, strength)

        # Staleness = inverse of retention, boosted by volatility
        # retention 1.0 → staleness 0.0 (fresh)
        # retention 0.0 → staleness 1.0 (stale)
        base_staleness = 1.0 - retention

        # Content age adds staleness if content dates are old
        content_retention = _ebbinghaus_retention(content_age, strength)
        content_staleness = 1.0 - content_retention

        staleness = (
            base_staleness * 0.4
            + content_staleness * 0.3
            + min(volatility, 1.0) * 0.3
        )

        if staleness < 0.5:
            continue

        severity = 'CRITICAL' if staleness > 1.0 else 'WARNING'
        msg = f'Staleness {staleness:.2f} (eff_age: {effective_age}d, content: {content_age}d'
        if access_count > 0:
            msg += f', accessed: {access_count}x'
        msg += ')'

        issues.append(Issue(
            severity=severity,
            category='stale',
            message=msg,
            files=[f.stem],
            details={
                'staleness_score': round(staleness, 3),
                'file_age_days': age_days,
                'effective_age_days': effective_age,
                'content_age_days': content_age,
                'volatility': round(volatility, 3),
                'markers': markers,
                'access_count': access_count,
                'retention': round(retention, 3),
                'type': parse_frontmatter(content)[0].get('type', 'unknown'),
            }
        ))

    return sorted(issues, key=lambda x: x.details.get('staleness_score', 0), reverse=True)


# --- 4. Orphan Detection ---

def find_orphans(
    memory_dir: Path,
) -> list[Issue]:
    """Find broken references to files and cross-refs that don't exist.

    No heavy dependencies — pure filesystem + regex.
    """
    issues = []

    FILE_PATH_RE = re.compile(r'(?:~/|/home/)[^\s\n\r\|`\'">\)]+')
    MEMORY_REF_RE = re.compile(r'`([a-z][a-z0-9_]+\.md)`')
    LINK_RE = re.compile(r'\[.*?\]\(([^)]+\.md)\)')
    SKIP_PATH = re.compile(r'[<>{}*?]|YYYY|MM-DD|\$\(')

    memory_stems = {f.stem for f in memory_dir.glob('*.md')}

    for f in sorted(memory_dir.glob('*.md')):
        if f.name in ('MEMORY.md', 'README.md'):
            continue

        content = f.read_text(encoding='utf-8', errors='replace')
        _, body = parse_frontmatter(content)

        for match in FILE_PATH_RE.finditer(body):
            ref = match.group(0).rstrip('.,;:)*')
            if SKIP_PATH.search(ref):
                continue
            expanded = Path(ref.replace('~', str(Path.home())))
            if not expanded.exists():
                issues.append(Issue(
                    severity='WARNING',
                    category='orphan',
                    message=f'Path does not exist: {ref}',
                    files=[f.stem],
                    details={
                        'missing_path': ref,
                        'context': _line_context(body, match.start()),
                    }
                ))

        refs = set()
        for match in MEMORY_REF_RE.finditer(body):
            refs.add(match.group(1))
        for match in LINK_RE.finditer(body):
            ref = match.group(1)
            if not ref.startswith('http'):
                refs.add(Path(ref).name)

        for ref in refs:
            stem = ref.replace('.md', '')
            if stem not in memory_stems and ref != f.name:
                issues.append(Issue(
                    severity='INFO',
                    category='orphan',
                    message=f'Memory cross-ref not found: {ref}',
                    files=[f.stem],
                    details={'missing_ref': ref}
                ))

    return issues


# --- 4b. Provenance Check (Source Monitoring) ---

REQUIRED_PROVENANCE = {'name', 'type', 'description'}


def find_unprovenanced(
    memory_dir: Path,
    required_fields: set[str] | None = None,
) -> list[Issue]:
    """Flag memory files missing provenance metadata.

    Brain analog: source monitoring failure → source amnesia.
    Files without metadata about their origin are untrusted —
    they are facts without provenance, vulnerable to confabulation.

    No heavy dependencies — pure filesystem + frontmatter parsing.
    """
    if required_fields is None:
        required_fields = REQUIRED_PROVENANCE

    issues = []

    for f in sorted(memory_dir.glob('*.md')):
        if f.name in ('MEMORY.md', 'README.md'):
            continue

        content = f.read_text(encoding='utf-8', errors='replace')
        meta, _ = parse_frontmatter(content)

        missing = []
        for field in sorted(required_fields):
            val = meta.get(field, '').strip()
            if not val:
                missing.append(field)

        if missing:
            severity = 'WARNING' if len(missing) < len(required_fields) else 'CRITICAL'
            issues.append(Issue(
                severity=severity,
                category='provenance',
                message=f'Missing provenance: {", ".join(missing)}',
                files=[f.stem],
                details={
                    'missing_fields': missing,
                    'has_frontmatter': bool(meta),
                    'fields_present': sorted(meta.keys()) if meta else [],
                }
            ))

    return sorted(issues, key=lambda x: len(x.details.get('missing_fields', [])), reverse=True)


# --- 5. Pre-Write Contradiction Check ---

def pre_write_check(
    new_text: str,
    store_dir: Path,
    source_file: str = '',
    top_k: int = 5,
    nli_threshold: float = 0.7,
) -> list[Issue]:
    """Check if new text contradicts existing memories BEFORE writing.

    Fast path: embeds new text, queries similar records, runs NLI.
    Call before writing any new memory or updating an existing one.
    """
    if not new_text or not new_text.strip():
        return []

    import numpy as np
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(store_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    model = SentenceTransformer(EMBED_MODEL)

    try:
        from sentence_transformers import CrossEncoder
        nli = CrossEncoder(NLI_MODEL)
    except Exception:
        return []

    chunks = []
    for line in new_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('---') or line.startswith('|'):
            continue
        if len(line) > 30:
            chunks.append(line)

    if not chunks:
        chunks = [new_text[:500]]

    issues = []
    seen = set()

    for chunk in chunks[:5]:
        embedding = model.encode([chunk], show_progress_bar=False).tolist()

        kwargs = {
            'query_embeddings': embedding,
            'n_results': top_k,
            'include': ['documents', 'metadatas', 'distances'],
        }
        if source_file:
            kwargs['where'] = {'source_file': {'$ne': source_file}}

        results = collection.query(**kwargs)

        for j in range(len(results['ids'][0])):
            distance = results['distances'][0][j]
            sim = 1 - (distance / 2)

            if sim < 0.5 or sim > 0.92:
                continue

            existing_text = results['documents'][0][j]
            existing_file = results['metadatas'][0][j].get('source_file', '')

            new_sent = _core_assertion(chunk)
            old_sent = _core_assertion(existing_text)

            logits = nli.predict([(new_sent, old_sent)])
            logits_arr = np.array(logits)
            if logits_arr.ndim == 1:
                logits_arr = logits_arr.reshape(1, -1)
            probs = _softmax(logits_arr)
            # DeBERTa NLI labels: [contradiction, neutral, entailment]
            c_prob = float(probs[0, 0])  # contradiction
            e_prob = float(probs[0, 2])  # entailment

            ent_new = _extract_entities(chunk)
            ent_old = _extract_entities(existing_text)

            key = (existing_file, chunk[:50])
            if key in seen:
                continue

            # Contradiction: high c_prob + shared entities
            if c_prob >= nli_threshold and (ent_new & ent_old):
                seen.add(key)
                shared = list(ent_new & ent_old)[:3]
                issues.append(Issue(
                    severity='CRITICAL' if c_prob > 0.9 else 'WARNING',
                    category='pre_write_conflict',
                    message=f'New text may contradict: {existing_file} (NLI={c_prob:.2f})',
                    files=[existing_file],
                    details={
                        'new_text': chunk[:200],
                        'existing_text': existing_text[:200],
                        'nli_score': round(c_prob, 3),
                        'similarity': round(sim, 3),
                        'shared_entities': shared,
                    }
                ))

            # Supersession: high similarity + high entailment = new version of same fact
            # Brain analog: retrieval-induced forgetting — accessing new version
            # suppresses the old version's accessibility
            elif e_prob > 0.6 and sim > 0.65 and (ent_new & ent_old):
                seen.add(key)
                shared = list(ent_new & ent_old)[:3]
                issues.append(Issue(
                    severity='INFO',
                    category='pre_write_supersession',
                    message=f'May supersede: {existing_file} (entail={e_prob:.2f}, sim={sim:.2f})',
                    files=[existing_file],
                    details={
                        'new_text': chunk[:200],
                        'existing_text': existing_text[:200],
                        'entailment_score': round(e_prob, 3),
                        'similarity': round(sim, 3),
                        'shared_entities': shared,
                        'existing_id': results['ids'][0][j],
                    }
                ))

    return sorted(issues, key=lambda x: x.details.get('nli_score',
                   x.details.get('entailment_score', 0)), reverse=True)[:10]


def apply_supersession_decay(
    issues: list[Issue],
    store_dir: Path,
    decay_factor: float = 0.7,
) -> int:
    """Decay health scores for superseded entries.

    Brain analog: retrieval-induced forgetting — when a newer version
    of a fact is accessed, the old version's accessibility decreases.

    Call after pre_write_check returns supersession issues and the
    new memory has been written.

    Returns number of records decayed.
    """
    supersessions = [i for i in issues if i.category == 'pre_write_supersession']
    if not supersessions:
        return 0

    import chromadb
    client = chromadb.PersistentClient(path=str(store_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return 0

    decayed = 0
    for issue in supersessions:
        source = issue.files[0] if issue.files else ''
        if not source:
            continue

        try:
            records = collection.get(
                where={'source_file': source},
                include=['metadatas'],
            )
            if not records['ids']:
                continue

            batch_ids = []
            batch_metas = []
            for i, rec_id in enumerate(records['ids']):
                meta = dict(records['metadatas'][i])
                old_health = float(meta.get('health_score', 1.0))
                meta['health_score'] = round(max(0.1, old_health * decay_factor), 3)
                meta['superseded_by'] = issue.details.get('new_text', '')[:100]
                batch_ids.append(rec_id)
                batch_metas.append(meta)

            collection.update(ids=batch_ids, metadatas=batch_metas)
            decayed += len(batch_ids)
        except Exception:
            continue

    return decayed


# --- 6. Health Scores ---

def compute_health_scores(results: dict[str, list[Issue]]) -> dict[str, float]:
    """Compute health score per source file from scan results.

    1.0 = healthy. Floor at 0.1 — unhealthy memories rank lower but never vanish.
    """
    scores = {}
    PENALTIES = {
        'contradictions': 0.2,
        'duplicates': 0.1,
        'stale': 0.25,
        'orphans': 0.15,
        'provenance': 0.15,
        'isolated': 0.1,
    }

    for category, issues_list in results.items():
        penalty = PENALTIES.get(category, 0.1)
        for issue in issues_list:
            for f in issue.files:
                if f not in scores:
                    scores[f] = 1.0
                if category == 'stale':
                    s = issue.details.get('staleness_score', 0.5)
                    scores[f] = max(0.1, scores[f] - s * penalty)
                else:
                    scores[f] = max(0.1, scores[f] - penalty)

    return scores


def update_health_scores(
    scores: dict[str, float],
    store_dir: Path,
) -> int:
    """Write health scores into ChromaDB metadata.

    Downstream RAG retrieval can read these to deprioritize unhealthy memories:
        final_score = semantic_similarity * health_score
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(store_dir))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return 0

    count = collection.count()
    if count == 0:
        return 0

    all_records = collection.get(include=['metadatas'], limit=count)

    batch_ids = []
    batch_metas = []

    for i, rec_id in enumerate(all_records['ids']):
        meta = all_records['metadatas'][i]
        source = meta.get('source_file', '')
        health = scores.get(source, 1.0)

        old_health = meta.get('health_score')
        if old_health is None or abs(float(old_health) - health) > 0.01:
            meta['health_score'] = round(health, 3)
            batch_ids.append(rec_id)
            batch_metas.append(meta)

    BATCH = 64
    for start in range(0, len(batch_ids), BATCH):
        collection.update(
            ids=batch_ids[start:start + BATCH],
            metadatas=batch_metas[start:start + BATCH],
        )

    return len(batch_ids)


# --- Full scan ---

def full_scan(
    memory_dir: Path,
    store_dir: Path | None = None,
    checks: list[str] | None = None,
) -> dict[str, list[Issue]]:
    """Run all health checks and return categorized issues."""
    if store_dir is None:
        store_dir = default_store_dir(memory_dir)
    if checks is None:
        checks = ['duplicates', 'isolated', 'orphans', 'stale', 'provenance', 'contradictions']

    results = {}
    if 'duplicates' in checks:
        results['duplicates'] = find_duplicates(store_dir)
    if 'isolated' in checks:
        results['isolated'] = find_isolated(store_dir)
    if 'orphans' in checks:
        results['orphans'] = find_orphans(memory_dir)
    if 'stale' in checks:
        results['stale'] = find_stale(memory_dir, store_dir=store_dir)
    if 'provenance' in checks:
        results['provenance'] = find_unprovenanced(memory_dir)
    if 'contradictions' in checks:
        results['contradictions'] = find_contradictions(store_dir)
    return results


# --- Report formatting ---

def format_report(results: dict[str, list[Issue]]) -> str:
    """Format scan results as a readable report."""
    lines = [
        '=' * 60,
        '  VIGIL — Memory Health Report',
        '=' * 60,
        '',
    ]

    total = sum(len(v) for v in results.values())
    critical = sum(1 for v in results.values() for i in v if i.severity == 'CRITICAL')
    warnings = sum(1 for v in results.values() for i in v if i.severity == 'WARNING')
    info = sum(1 for v in results.values() for i in v if i.severity == 'INFO')

    if total == 0:
        lines.append('  CLEAN — no health issues detected.')
        return '\n'.join(lines)

    lines.append(f'  {total} issues: {critical} CRITICAL / {warnings} WARNING / {info} INFO')
    lines.append('')

    for category in ['contradictions', 'duplicates', 'stale', 'orphans']:
        issues = results.get(category, [])
        if not issues:
            continue

        lines.append(f'--- {category.upper()} ({len(issues)}) ---')
        lines.append('')

        for issue in issues:
            icon = {'CRITICAL': '!!!', 'WARNING': ' ! ', 'INFO': ' i '}[issue.severity]
            lines.append(f'  [{icon}] {issue.message}')
            lines.append(f'         files: {", ".join(issue.files)}')

            if issue.category in ('contradiction', 'duplicate'):
                lines.append(f'         A: {issue.details.get("text_a", "")[:120]}')
                lines.append(f'         B: {issue.details.get("text_b", "")[:120]}')
            elif issue.category == 'stale':
                m = issue.details.get('markers', [])
                if m:
                    lines.append(f'         markers: {", ".join(m)}')
            elif issue.category == 'orphan':
                if 'missing_path' in issue.details:
                    lines.append(f'         path: {issue.details["missing_path"]}')
                elif 'missing_ref' in issue.details:
                    lines.append(f'         ref: {issue.details["missing_ref"]}')
            lines.append('')

    return '\n'.join(lines)
