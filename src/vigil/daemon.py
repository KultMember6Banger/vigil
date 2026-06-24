"""`vigil serve` — a long-lived pre-write check daemon.

Loads the embedder + NLI model ONCE and answers pre-write checks over a local
unix-domain socket using newline-delimited JSON (one request object per line,
one response object per line). This gives the git hook / agent sub-second checks
instead of paying the model-load cost (seconds) on every call.

Protocol (JSON lines over the socket):
  request : {"text": "...", "source": "", "collection": null}
  response: {"ok": true, "conflict_count": N, "conflicts": [issue, ...]}
            {"ok": false, "error": "..."}

The socket path defaults to `<store_dir>/vigil.sock`. The client helper
`check_via_daemon` connects if the socket exists, else signals the caller to
fall back to an in-process check (so `vigil check --daemon` degrades gracefully).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

SOCK_NAME = 'vigil.sock'
_RECV = 65536
# AF_UNIX paths are capped by the OS (~104 chars on macOS, ~108 on Linux). When
# the in-store path would exceed this, fall back to a short hashed name in the
# system temp dir so both serve() and the client agree on the same location.
_SOCK_MAX = 100


def socket_path(store_dir: Path) -> Path:
    """Default daemon socket path.

    Normally `<store_dir>/vigil.sock`. If that absolute path is too long for an
    AF_UNIX socket, deterministically relocate it to a short name under the
    system temp dir (derived from a hash of the store path) so the daemon and
    its clients still resolve to the same socket.
    """
    candidate = Path(store_dir).resolve() / SOCK_NAME
    if len(str(candidate)) <= _SOCK_MAX:
        return candidate
    digest = hashlib.sha1(str(Path(store_dir).resolve()).encode('utf-8')).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f'vigil-{digest}.sock'


def _issue_to_dict(issue) -> dict:
    return {
        'severity': issue.severity,
        'category': issue.category,
        'message': issue.message,
        'files': issue.files,
        'details': issue.details,
    }


def serve(
    store_dir: Path,
    collection_name: str | None = None,
    sock_path: Path | None = None,
    log=None,
) -> None:
    """Run the pre-write daemon until interrupted.

    Loads models once (warm-up), binds the unix socket, then serves one
    connection at a time. Cleans up the socket file on exit.
    """
    from .scanner import pre_write_check, _load_embedder, _load_nli, EMBED_MODEL, NLI_MODEL

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    sock_path = Path(sock_path) if sock_path else socket_path(store_dir)

    # Warm the models once so every request is a fast path.
    log(f'vigil serve: loading models ({EMBED_MODEL} + {NLI_MODEL})...')
    _load_embedder(EMBED_MODEL)
    try:
        _load_nli(NLI_MODEL)
    except Exception as e:
        log(f'vigil serve: WARN NLI unavailable ({e})')

    if sock_path.exists():
        sock_path.unlink()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    log(f'vigil serve: listening on {sock_path} (collection='
        f'{collection_name or "<default>"})')

    try:
        while True:
            conn, _ = srv.accept()
            try:
                _handle_conn(conn, store_dir, collection_name, pre_write_check, log)
            finally:
                conn.close()
    except KeyboardInterrupt:
        log('vigil serve: shutting down')
    finally:
        srv.close()
        try:
            sock_path.unlink()
        except OSError:
            pass


def _handle_conn(conn, store_dir, collection_name, pre_write_check, log):
    """Serve newline-delimited JSON requests on a single connection."""
    buf = b''
    while True:
        chunk = conn.recv(_RECV)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            line = line.strip()
            if not line:
                continue
            resp = _process_line(line, store_dir, collection_name, pre_write_check, log)
            conn.sendall((json.dumps(resp) + '\n').encode('utf-8'))


def _process_line(line: bytes, store_dir, collection_name, pre_write_check, log) -> dict:
    try:
        req = json.loads(line.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {'ok': False, 'error': f'bad request: {e}'}

    if req.get('command') == 'ping':
        return {'ok': True, 'pong': True}

    text = req.get('text', '')
    source = req.get('source', '')
    coll = req.get('collection') or collection_name
    try:
        issues = pre_write_check(
            text, store_dir=store_dir, source_file=source, collection_name=coll)
    except Exception as e:  # pragma: no cover - defensive
        log(f'vigil serve: check error: {e}')
        return {'ok': False, 'error': str(e)}

    return {
        'ok': True,
        'conflict_count': len(issues),
        'conflicts': [_issue_to_dict(i) for i in issues],
    }


# --- client ---

def daemon_running(store_dir: Path, sock_path: Path | None = None) -> bool:
    """True if a daemon socket exists and accepts a ping."""
    sock_path = Path(sock_path) if sock_path else socket_path(store_dir)
    if not sock_path.exists():
        return False
    try:
        resp = _send(sock_path, {'command': 'ping'})
        return bool(resp.get('ok'))
    except OSError:
        return False


def check_via_daemon(
    store_dir: Path,
    text: str,
    source: str = '',
    collection_name: str | None = None,
    sock_path: Path | None = None,
) -> dict | None:
    """Run a pre-write check through the daemon.

    Returns the response dict, or None if the daemon is not reachable (caller
    should then fall back to an in-process check).
    """
    sock_path = Path(sock_path) if sock_path else socket_path(store_dir)
    if not sock_path.exists():
        return None
    try:
        return _send(sock_path, {
            'text': text, 'source': source, 'collection': collection_name})
    except OSError:
        return None


def _send(sock_path: Path, payload: dict) -> dict:
    """Send one JSON request, read one JSON response line."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30)
    try:
        client.connect(str(sock_path))
        client.sendall((json.dumps(payload) + '\n').encode('utf-8'))
        buf = b''
        while b'\n' not in buf:
            chunk = client.recv(_RECV)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b'\n', 1)[0]
        return json.loads(line.decode('utf-8'))
    finally:
        client.close()
