"""Tests for the `vigil hook` pre-commit gate (pre_write_check mocked)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil import hook
from vigil.scanner import Issue


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def test_install_standalone_when_no_git(tmp_path):
    # tmp_path is not a git repo -> standalone script under .vigil/
    with mock.patch.object(hook, "_git_repo_root", return_value=None):
        info = hook.install_hook(tmp_path)
    assert info["mode"] == "standalone"
    p = Path(info["path"])
    assert p.exists()
    assert os.access(p, os.X_OK)
    assert "vigil hook run" in p.read_text()


def test_install_git_repo(tmp_path):
    if not _git(tmp_path, "init").returncode == 0:
        pytest.skip("git not available")
    info = hook.install_hook(tmp_path)
    assert info["mode"] == "git"
    p = Path(info["path"])
    assert p.name == "pre-commit"
    assert os.access(p, os.X_OK)


def test_install_git_repo_chains_existing(tmp_path):
    if not _git(tmp_path, "init").returncode == 0:
        pytest.skip("git not available")
    hooks = tmp_path / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    existing = hooks / "pre-commit"
    existing.write_text("#!/bin/sh\necho existing\n")
    info = hook.install_hook(tmp_path)
    assert info["chained"] is True
    body = existing.read_text()
    assert "echo existing" in body          # original preserved
    assert "Vigil gate" in body             # vigil chained in


def _clear_issue(*a, **k):
    return []


def _critical_issue(*a, **k):
    return [Issue("CRITICAL", "pre_write_conflict", "contradicts foo.md",
                  files=["foo"], details={})]


def _warning_issue(*a, **k):
    return [Issue("WARNING", "pre_write_conflict", "maybe contradicts",
                  files=["foo"], details={})]


def test_run_hook_blocks_on_critical(tmp_path):
    (tmp_path / "a.md").write_text("some new memory")
    out = io.StringIO()
    with mock.patch.object(hook, "_git_repo_root", return_value=None), \
         mock.patch("vigil.scanner.pre_write_check", side_effect=_critical_issue), \
         mock.patch("vigil.daemon.daemon_running", return_value=False):
        code = hook.run_hook(tmp_path, store_dir=tmp_path / ".vigil", out=out)
    assert code == 1
    assert "BLOCKED" in out.getvalue()


def test_run_hook_warns_but_allows(tmp_path):
    (tmp_path / "a.md").write_text("some new memory")
    out = io.StringIO()
    with mock.patch.object(hook, "_git_repo_root", return_value=None), \
         mock.patch("vigil.scanner.pre_write_check", side_effect=_warning_issue), \
         mock.patch("vigil.daemon.daemon_running", return_value=False):
        code = hook.run_hook(tmp_path, store_dir=tmp_path / ".vigil", out=out)
    assert code == 0
    assert "WARNING" in out.getvalue()


def test_run_hook_clear_passes(tmp_path):
    (tmp_path / "a.md").write_text("some new memory")
    out = io.StringIO()
    with mock.patch.object(hook, "_git_repo_root", return_value=None), \
         mock.patch("vigil.scanner.pre_write_check", side_effect=_clear_issue), \
         mock.patch("vigil.daemon.daemon_running", return_value=False):
        code = hook.run_hook(tmp_path, store_dir=tmp_path / ".vigil", out=out)
    assert code == 0


def test_run_hook_no_files_passes(tmp_path):
    out = io.StringIO()
    with mock.patch.object(hook, "_git_repo_root", return_value=None):
        code = hook.run_hook(tmp_path, store_dir=tmp_path / ".vigil", out=out)
    assert code == 0
