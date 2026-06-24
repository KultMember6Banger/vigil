"""Tests for the `vigil serve` daemon — models mocked, real unix socket.

We patch the model loaders + pre_write_check so the daemon serves instantly,
then exercise the real socket protocol (ping, check, fallback when no daemon).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil import daemon
from vigil.scanner import Issue


def _fake_check(text, store_dir=None, source_file="", collection_name=None):
    if "conflict" in text:
        return [Issue("CRITICAL", "pre_write_conflict", "boom",
                      files=["x"], details={"new_text": text})]
    return []


@pytest.fixture
def running_daemon(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    # Let the daemon pick the socket path (it relocates to a short temp path
    # when the in-store path is too long for AF_UNIX, e.g. deep pytest tmpdirs).
    sock = daemon.socket_path(store)
    # Clear any stale socket from a prior run that hashed to the same temp name.
    if sock.exists():
        sock.unlink()

    # serve() captures pre_write_check + the model loaders at start; patch them
    # so the daemon comes up instantly with the fake checker baked in.
    with mock.patch("vigil.scanner.pre_write_check", side_effect=_fake_check), \
         mock.patch("vigil.scanner._load_embedder", return_value=object()), \
         mock.patch("vigil.scanner._load_nli", return_value=object()):
        t = threading.Thread(
            target=daemon.serve,
            kwargs={"store_dir": store, "log": lambda m: None},
            daemon=True,
        )
        t.start()
        # Poll until the daemon actually answers a ping (not just file exists).
        ready = False
        for _ in range(200):
            if sock.exists() and daemon.daemon_running(store, sock):
                ready = True
                break
            time.sleep(0.02)
        assert ready, "daemon never became ready"
        yield store, sock
        # thread is daemon=True and dies with the test process


def test_daemon_ping(running_daemon):
    store, sock = running_daemon
    assert daemon.daemon_running(store, sock) is True


def test_daemon_check_clear(running_daemon):
    store, sock = running_daemon
    resp = daemon.check_via_daemon(store, "all good here", sock_path=sock)
    assert resp is not None
    assert resp["ok"] is True
    assert resp["conflict_count"] == 0


def test_daemon_check_conflict(running_daemon):
    store, sock = running_daemon
    resp = daemon.check_via_daemon(store, "this is a conflict", sock_path=sock)
    assert resp["ok"] is True
    assert resp["conflict_count"] == 1
    assert resp["conflicts"][0]["severity"] == "CRITICAL"


def test_check_via_daemon_returns_none_when_absent(tmp_path):
    # No socket exists -> client signals fallback (None).
    assert daemon.check_via_daemon(tmp_path, "x") is None
    assert daemon.daemon_running(tmp_path) is False
