"""Vigil configuration — optional `.vigil.toml` overrides for all checks.

Precedence (highest wins): CLI flags > `.vigil.toml` > built-in defaults.

Loading order: tomllib (Python 3.11+) is used when available; otherwise a tiny
dependency-free fallback parser handles the limited TOML subset Vigil needs
(top-level key/value pairs and one level of `[section]` tables, with strings,
ints, floats, booleans and flat arrays). The fallback is intentionally small —
it is not a general TOML implementation, only enough for `.vigil.toml`.

Every configurable knob has a built-in default so a missing or partial config
file always yields a complete, usable VigilConfig.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path


# --- Built-in defaults (mirror the historical hard-coded values) ---

DEFAULT_PROVENANCE_FIELDS = ['name', 'type', 'description']
DEFAULT_VOLATILITY_MARKERS = [
    {'pattern': r'\b(current|currently|ongoing|in.?progress|active)\b',
     'label': 'temporal_state', 'weight': 0.3},
    {'pattern': r'\b(pending|waiting|queued|blocked|next)\b',
     'label': 'pending_action', 'weight': 0.2},
    {'pattern': r'\bstatus:\s*\w+', 'label': 'explicit_status', 'weight': 0.3},
    {'pattern': r'\b(todo|TODO|FIXME|HACK|TEMP)\b',
     'label': 'action_marker', 'weight': 0.2},
]


@dataclass
class VigilConfig:
    """Resolved configuration for a scan/check run.

    Construct with VigilConfig.load(memory_dir) (or .from_dict / defaults), then
    let CLI flags override individual fields via .override(...).
    """

    # provenance
    required_provenance_fields: list = field(
        default_factory=lambda: list(DEFAULT_PROVENANCE_FIELDS))

    # contradictions
    sim_low: float = 0.65
    sim_high: float = 0.90
    nli_threshold: float = 0.85

    # isolated
    isolation_threshold: float = 0.3

    # staleness
    warn_days: int = 14
    critical_days: int = 30
    staleness_base_weight: float = 0.4
    staleness_content_weight: float = 0.3
    staleness_volatility_weight: float = 0.3
    volatility_markers: list = field(
        default_factory=lambda: [dict(m) for m in DEFAULT_VOLATILITY_MARKERS])

    # duplicates
    duplicate_threshold: float = 0.85

    # fix / archive
    archive_dir: str = 'archive'
    fix_duplicate_threshold: float = 0.92  # conservative auto-archive gate

    # ignore globs applied across all checks
    ignore_globs: list = field(default_factory=list)

    # provenance for source: which file did this config come from (info only)
    source_path: str = ''

    # --- construction ---

    @classmethod
    def defaults(cls) -> 'VigilConfig':
        return cls()

    @classmethod
    def from_dict(cls, data: dict, source_path: str = '') -> 'VigilConfig':
        """Build a config from a parsed TOML dict.

        Recognized layout (all optional):

            required_provenance_fields = ["name", "type", "description"]
            ignore_globs = ["scratch/*", "*.draft.md"]

            [contradictions]
            sim_low = 0.65
            sim_high = 0.90
            nli_threshold = 0.85

            [isolated]
            isolation_threshold = 0.3

            [staleness]
            warn_days = 14
            critical_days = 30
            base_weight = 0.4
            content_weight = 0.3
            volatility_weight = 0.3
            [[staleness.volatility_markers]]
            pattern = "..."; label = "..."; weight = 0.3

            [duplicates]
            threshold = 0.85

            [fix]
            archive_dir = "archive"
            duplicate_threshold = 0.92
        """
        cfg = cls()
        cfg.source_path = source_path

        if 'required_provenance_fields' in data:
            cfg.required_provenance_fields = [
                str(x) for x in data['required_provenance_fields']]
        if 'ignore_globs' in data:
            cfg.ignore_globs = [str(x) for x in data['ignore_globs']]

        contr = data.get('contradictions', {})
        if isinstance(contr, dict):
            cfg.sim_low = float(contr.get('sim_low', cfg.sim_low))
            cfg.sim_high = float(contr.get('sim_high', cfg.sim_high))
            cfg.nli_threshold = float(contr.get('nli_threshold', cfg.nli_threshold))

        iso = data.get('isolated', {})
        if isinstance(iso, dict):
            cfg.isolation_threshold = float(
                iso.get('isolation_threshold', cfg.isolation_threshold))

        stale = data.get('staleness', {})
        if isinstance(stale, dict):
            cfg.warn_days = int(stale.get('warn_days', cfg.warn_days))
            cfg.critical_days = int(stale.get('critical_days', cfg.critical_days))
            cfg.staleness_base_weight = float(
                stale.get('base_weight', cfg.staleness_base_weight))
            cfg.staleness_content_weight = float(
                stale.get('content_weight', cfg.staleness_content_weight))
            cfg.staleness_volatility_weight = float(
                stale.get('volatility_weight', cfg.staleness_volatility_weight))
            markers = stale.get('volatility_markers')
            if isinstance(markers, list) and markers:
                parsed = []
                for m in markers:
                    if not isinstance(m, dict):
                        continue
                    parsed.append({
                        'pattern': str(m.get('pattern', '')),
                        'label': str(m.get('label', 'marker')),
                        'weight': float(m.get('weight', 0.2)),
                    })
                if parsed:
                    cfg.volatility_markers = parsed

        dup = data.get('duplicates', {})
        if isinstance(dup, dict):
            cfg.duplicate_threshold = float(
                dup.get('threshold', cfg.duplicate_threshold))

        fix = data.get('fix', {})
        if isinstance(fix, dict):
            cfg.archive_dir = str(fix.get('archive_dir', cfg.archive_dir))
            cfg.fix_duplicate_threshold = float(
                fix.get('duplicate_threshold', cfg.fix_duplicate_threshold))

        return cfg

    @classmethod
    def load(cls, memory_dir: Path | None = None,
             config_path: Path | None = None) -> 'VigilConfig':
        """Load `.vigil.toml`.

        If config_path is given, load it (missing file -> defaults). Else look
        for `<memory_dir>/.vigil.toml`. A missing file is not an error; it just
        yields built-in defaults.
        """
        path = None
        if config_path is not None:
            path = Path(config_path)
        elif memory_dir is not None:
            cand = Path(memory_dir) / '.vigil.toml'
            if cand.exists():
                path = cand

        if path is None or not path.exists():
            return cls.defaults()

        data = parse_toml(path.read_text(encoding='utf-8'))
        return cls.from_dict(data, source_path=str(path))

    # --- overrides ---

    def override(self, **kwargs) -> 'VigilConfig':
        """Return a copy with the given non-None fields overridden.

        Used to layer CLI flags on top of file/default config. None values are
        ignored so unset flags do not clobber configured/default values.
        """
        valid = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in valid:
                raise KeyError(f'unknown config field: {k}')
            setattr(self, k, v)
        return self


# --- TOML parsing ---

def parse_toml(text: str) -> dict:
    """Parse TOML using stdlib tomllib when available, else the fallback."""
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # optional backport for <3.11
        return tomli.loads(text)
    except ModuleNotFoundError:
        pass
    return _fallback_parse_toml(text)


_INT_RE = re.compile(r'^[+-]?\d+$')
_FLOAT_RE = re.compile(r'^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$')


_BASIC_ESCAPES = {
    '\\': '\\', '"': '"', 'b': '\b', 'f': '\f', 'n': '\n',
    'r': '\r', 't': '\t',
}


def _unescape_basic(s: str) -> str:
    """Apply TOML basic-string escape sequences (so `\\b` -> a single backslash
    then `b`, matching tomllib). Unknown escapes are left as-is (backslash kept).
    """
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in _BASIC_ESCAPES:
                out.append(_BASIC_ESCAPES[nxt])
                i += 2
                continue
            if nxt in ('u', 'U'):
                width = 4 if nxt == 'u' else 8
                hexpart = s[i + 2:i + 2 + width]
                try:
                    out.append(chr(int(hexpart, 16)))
                    i += 2 + width
                    continue
                except ValueError:
                    pass
            # unknown escape: keep the backslash literally
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _coerce_scalar(tok: str):
    tok = tok.strip()
    if not tok:
        return ''
    if tok[0] == '"' and tok[-1] == '"' and len(tok) >= 2:
        return _unescape_basic(tok[1:-1])
    if tok[0] == "'" and tok[-1] == "'" and len(tok) >= 2:
        # literal string — no escapes
        return tok[1:-1]
    if tok in ('true', 'false'):
        return tok == 'true'
    if _INT_RE.match(tok):
        return int(tok)
    if _FLOAT_RE.match(tok) and ('.' in tok or 'e' in tok or 'E' in tok):
        return float(tok)
    return tok


def _split_array(inner: str) -> list:
    """Split a flat (non-nested) inline array body on commas, honoring quotes."""
    items = []
    cur = ''
    quote = None
    for ch in inner:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            cur += ch
        elif ch == ',':
            if cur.strip():
                items.append(_coerce_scalar(cur))
            cur = ''
        else:
            cur += ch
    if cur.strip():
        items.append(_coerce_scalar(cur))
    return items


def _strip_comment(line: str) -> str:
    """Remove a trailing # comment not inside a string."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == '#':
            return line[:i]
    return line


def _fallback_parse_toml(text: str) -> dict:
    """A tiny TOML subset parser (no external deps, Python 3.9-safe).

    Supports: top-level scalars/arrays, `[section]` tables (one level), and
    `[[section.array]]` arrays-of-tables (one nesting level, used for
    volatility_markers). Values: strings, ints, floats, bools, flat arrays.
    Multiline arrays and nested inline tables are NOT supported.
    """
    root: dict = {}
    # current insertion target for plain key=value lines
    target = root
    # for [[a.b]] array-of-tables we track the table dict so following keys land
    array_table = None

    pending_array_key = None  # multi-line array accumulation
    pending_array_buf = ''

    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        stripped = line.strip()

        if pending_array_key is not None:
            pending_array_buf += ' ' + stripped
            if ']' in stripped:
                inner = pending_array_buf[pending_array_buf.index('[') + 1:
                                          pending_array_buf.rindex(']')]
                target[pending_array_key] = _split_array(inner)
                pending_array_key = None
                pending_array_buf = ''
            continue

        if not stripped:
            continue

        # [[section.sub]] array-of-tables
        m = re.match(r'^\[\[\s*([A-Za-z0-9_.\-]+)\s*\]\]$', stripped)
        if m:
            path = m.group(1).split('.')
            node = root
            for part in path[:-1]:
                node = node.setdefault(part, {})
            arr = node.setdefault(path[-1], [])
            if not isinstance(arr, list):
                arr = []
                node[path[-1]] = arr
            array_table = {}
            arr.append(array_table)
            target = array_table
            continue

        # [section] table
        m = re.match(r'^\[\s*([A-Za-z0-9_.\-]+)\s*\]$', stripped)
        if m:
            path = m.group(1).split('.')
            node = root
            for part in path:
                node = node.setdefault(part, {})
            target = node
            array_table = None
            continue

        # key = value
        if '=' not in stripped:
            continue
        key, _, val = stripped.partition('=')
        key = key.strip().strip('"').strip("'")
        val = val.strip()

        if val.startswith('['):
            if ']' in val:
                inner = val[val.index('[') + 1:val.rindex(']')]
                target[key] = _split_array(inner)
            else:
                pending_array_key = key
                pending_array_buf = val
            continue

        target[key] = _coerce_scalar(val)

    return root
