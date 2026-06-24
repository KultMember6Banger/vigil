"""Vigil CLI — memory health monitor for AI agents.

Usage:
  vigil index   <memory_dir>              # Build/update search index
  vigil scan    <memory_dir>              # Full health scan
  vigil scan    <memory_dir> --check stale --check orphans  # Selective
  vigil check   <memory_dir> "new text"   # Pre-write contradiction check
  vigil check   <memory_dir> --file x.md  # Check file contents
  vigil check   <memory_dir> "t" --daemon # Use a running `vigil serve` daemon
  vigil health  <memory_dir>              # Full scan + update + history snapshot
  vigil fix     <memory_dir>              # Resolution plan (dry-run)
  vigil fix     <memory_dir> --apply --yes  # Apply (archive only; never deletes)
  vigil history <memory_dir>              # Health trend over time
  vigil serve   <memory_dir>              # Long-lived pre-write check daemon
  vigil hook    install <memory_dir>      # Install git pre-commit gate
  vigil hook    run <memory_dir>          # Hook body (run by the pre-commit hook)

Store dir resolves to --store, else $MEMORY_STORE, else <memory_dir>/.vigil/.
Collection resolves to --collection, else $MEMORY_COLLECTION, else 'agent_memory'.
Optional `<memory_dir>/.vigil.toml` (or --config) tunes thresholds; CLI flags
override config, config overrides built-in defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _resolve_store(args, memory_dir):
    """--store override, else env-aware default (MEMORY_STORE / .vigil)."""
    from .indexer import default_store_dir
    return Path(args.store) if getattr(args, 'store', None) else default_store_dir(memory_dir)


def _load_config(args, memory_dir):
    """Load `.vigil.toml` (or --config) and layer CLI flag overrides on top."""
    from .config import VigilConfig
    cfg = VigilConfig.load(
        memory_dir, config_path=Path(args.config) if getattr(args, 'config', None) else None)
    return cfg


def cmd_index(args):
    from .indexer import build_index, default_store_dir

    memory_dir = Path(args.memory_dir)
    store_dir = Path(args.store) if args.store else default_store_dir(memory_dir)

    print(f'Vigil indexing {memory_dir}...')
    stats = build_index(
        memory_dir, store_dir=store_dir,
        full_rebuild=args.rebuild,
        collection_name=args.collection,
    )
    print(f'  Indexed: {stats["indexed"]} files'
          f' (skipped {stats["skipped"]} unchanged)')
    print(f'  Records: {stats["records"]} total in store')
    print(f'  Time:    {stats["elapsed"]}s')


def cmd_scan(args):
    from .scanner import format_report, full_scan

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    coll = args.collection
    config = _load_config(args, memory_dir)

    checks = args.checks or ['duplicates', 'isolated', 'orphans', 'stale', 'provenance', 'contradictions']

    if not args.json:
        print(f'Vigil scanning {memory_dir}...')
    t0 = time.time()
    results = full_scan(memory_dir=memory_dir, store_dir=store_dir,
                        checks=checks, collection_name=coll, config=config)
    elapsed = time.time() - t0

    if args.json:
        out = {}
        for cat, issues_list in results.items():
            out[cat] = [{'severity': i.severity, 'message': i.message,
                         'files': i.files, 'details': i.details}
                        for i in issues_list]
        print(json.dumps(out, indent=2))
    else:
        print()
        print(format_report(results))
        print(f'Scan completed in {elapsed:.1f}s')


def cmd_check(args):
    from .scanner import pre_write_check

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)

    if args.file:
        text = Path(args.file).read_text(encoding='utf-8')
        source = Path(args.file).stem if not args.source else args.source
    elif args.text:
        text = args.text
        source = args.source or ''
    else:
        print('Error: provide text or --file')
        sys.exit(1)

    print('Vigil pre-write check...')
    t0 = time.time()

    issues = None
    used_daemon = False
    if getattr(args, 'daemon', False):
        from . import daemon
        resp = daemon.check_via_daemon(
            store_dir, text, source=source, collection_name=args.collection)
        if resp is not None and resp.get('ok'):
            issues = resp['conflicts']
            used_daemon = True
        else:
            print('  (daemon not reachable — falling back to in-process check)')

    if issues is None:
        raw = pre_write_check(text, store_dir=store_dir, source_file=source,
                              collection_name=args.collection)
        issues = [{'severity': i.severity, 'message': i.message,
                   'category': i.category, 'details': i.details} for i in raw]

    elapsed = time.time() - t0
    tag = ' via daemon' if used_daemon else ''

    if not issues:
        print(f'  CLEAR — no contradictions found ({elapsed:.1f}s{tag})')
    else:
        print(f'\n  {len(issues)} potential conflicts found ({elapsed:.1f}s{tag}):')
        print()
        for issue in issues:
            sev = issue['severity']
            icon = '!!!' if sev == 'CRITICAL' else ' ! '
            print(f'  [{icon}] {issue["message"]}')
            det = issue.get('details', {})
            if det.get('new_text'):
                print(f'         new:      {det["new_text"][:120]}')
            if det.get('existing_text'):
                print(f'         existing: {det["existing_text"][:120]}')
            print()


def cmd_health(args):
    from .scanner import full_scan, compute_health_scores, update_health_scores
    from .history import record_health_snapshot

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    coll = args.collection
    config = _load_config(args, memory_dir)
    now = datetime.now()

    print('Vigil health scan + score update...')
    t0 = time.time()

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir,
                        collection_name=coll, config=config, now=now)
    scores = compute_health_scores(results)
    n = update_health_scores(scores, store_dir, collection_name=coll)

    elapsed = time.time() - t0
    total_issues = sum(len(v) for v in results.values())

    print(f'\n  {total_issues} issues found, {n} records scored ({elapsed:.1f}s)')
    print()

    all_files = {f.stem for f in memory_dir.glob('*.md')
                 if f.name not in ('MEMORY.md', 'README.md')}
    for f in sorted(all_files):
        s = scores.get(f, 1.0)
        if s >= 1.0:
            continue
        icon = '!!!' if s < 0.5 else ' ! ' if s < 0.8 else ' . '
        print(f'  [{icon}] {f}: {s:.2f}')

    healthy = len(all_files) - len(scores)
    print(f'\n  {healthy} files healthy (1.00), {len(scores)} files with issues')

    # Record a history snapshot (full per-file picture, including 1.0 files).
    full_scores = {f: scores.get(f, 1.0) for f in all_files}
    totals = {'files': len(all_files),
              'with_issues': len(scores),
              'issues': total_issues}
    hist_path = record_health_snapshot(
        store_dir, full_scores, timestamp=now.isoformat(timespec='seconds'),
        totals=totals)
    print(f'\n  Snapshot appended to {hist_path}')


def cmd_fix(args):
    from .fix import run_fix, format_plan, plan_to_dict

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    coll = args.collection
    config = _load_config(args, memory_dir)

    apply = args.apply
    if apply and not args.yes:
        # Build a dry-run plan first to show what WOULD happen, then confirm.
        actions, _ = run_fix(memory_dir, store_dir, config,
                             collection_name=coll, apply=False)
        n_archive = sum(1 for a in actions if a.action == 'archive')
        print(format_plan(actions, applied=False, archive_dir_name=config.archive_dir))
        if n_archive == 0:
            print('\nNothing to apply.')
            return
        ans = input(f'\nArchive {n_archive} file(s) into {config.archive_dir}/? [y/N] ')
        if ans.strip().lower() not in ('y', 'yes'):
            print('Aborted.')
            return

    if not args.json:
        print(f'Vigil fix {memory_dir} ({"APPLY" if apply else "dry-run"})...')
    actions, _ = run_fix(memory_dir, store_dir, config,
                         collection_name=coll, apply=apply)

    if args.json:
        print(json.dumps(plan_to_dict(actions, applied=apply), indent=2))
    else:
        print()
        print(format_plan(actions, applied=apply, archive_dir_name=config.archive_dir))


def cmd_history(args):
    from .history import load_history, format_history

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)

    snapshots = load_history(store_dir)
    if args.json:
        print(json.dumps(snapshots, indent=2))
    else:
        print(format_history(snapshots))


def cmd_serve(args):
    from . import daemon

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    sock = Path(args.socket) if args.socket else None
    daemon.serve(store_dir, collection_name=args.collection, sock_path=sock)


def cmd_hook(args):
    from . import hook

    memory_dir = Path(args.memory_dir)
    if args.hook_action == 'install':
        info = hook.install_hook(memory_dir)
        print(f'Vigil hook installed ({info["mode"]}): {info["path"]}')
        if info.get('chained'):
            print(f'  chained into existing pre-commit: {info["pre_commit"]}')
        if info['mode'] == 'standalone':
            print('  (no git repo found — wire this script up as your pre-commit hook)')
    elif args.hook_action == 'run':
        store_dir = _resolve_store(args, memory_dir)
        code = hook.run_hook(memory_dir, store_dir=store_dir,
                             collection_name=args.collection)
        sys.exit(code)


def main():
    parser = argparse.ArgumentParser(
        description='Vigil — Memory health monitor for AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', help='Commands')

    coll_help = ('ChromaDB collection name (default: $MEMORY_COLLECTION '
                 "or 'agent_memory')")
    store_help = 'ChromaDB store path (default: $MEMORY_STORE or <memory_dir>/.vigil/)'
    config_help = 'Path to .vigil.toml (default: <memory_dir>/.vigil.toml if present)'

    # index
    p_idx = sub.add_parser('index', help='Build/update search index')
    p_idx.add_argument('memory_dir', help='Directory of markdown memory files')
    p_idx.add_argument('--store', help=store_help)
    p_idx.add_argument('--collection', help=coll_help)
    p_idx.add_argument('--rebuild', action='store_true', help='Full rebuild (delete existing index)')

    # scan
    p_scan = sub.add_parser('scan', help='Full health scan')
    p_scan.add_argument('memory_dir', help='Directory of markdown memory files')
    p_scan.add_argument('--check',
                        choices=['contradictions', 'duplicates', 'isolated', 'stale', 'orphans', 'provenance'],
                        action='append', dest='checks')
    p_scan.add_argument('--json', action='store_true', help='Output as JSON')
    p_scan.add_argument('--store', help=store_help)
    p_scan.add_argument('--collection', help=coll_help)
    p_scan.add_argument('--config', help=config_help)

    # check (pre-write)
    p_check = sub.add_parser('check', help='Pre-write contradiction check')
    p_check.add_argument('memory_dir', help='Directory of markdown memory files')
    p_check.add_argument('text', nargs='?', help='Text to check')
    p_check.add_argument('--file', help='Read text from file instead')
    p_check.add_argument('--source', default='', help='Source file stem to exclude')
    p_check.add_argument('--daemon', action='store_true',
                         help='Use a running `vigil serve` daemon if available')
    p_check.add_argument('--store', help=store_help)
    p_check.add_argument('--collection', help=coll_help)

    # health
    p_health = sub.add_parser('health', help='Full scan + update health scores + snapshot')
    p_health.add_argument('memory_dir', help='Directory of markdown memory files')
    p_health.add_argument('--store', help=store_help)
    p_health.add_argument('--collection', help=coll_help)
    p_health.add_argument('--config', help=config_help)

    # fix
    p_fix = sub.add_parser('fix', help='Build a resolution plan (dry-run) and optionally apply')
    p_fix.add_argument('memory_dir', help='Directory of markdown memory files')
    p_fix.add_argument('--apply', action='store_true',
                       help='Apply the plan (archive only — never deletes)')
    p_fix.add_argument('--yes', action='store_true',
                       help='Skip the confirmation prompt when applying')
    p_fix.add_argument('--json', action='store_true', help='Output the plan as JSON')
    p_fix.add_argument('--store', help=store_help)
    p_fix.add_argument('--collection', help=coll_help)
    p_fix.add_argument('--config', help=config_help)

    # history
    p_hist = sub.add_parser('history', help='Show health trend over time')
    p_hist.add_argument('memory_dir', help='Directory of markdown memory files')
    p_hist.add_argument('--json', action='store_true', help='Output snapshots as JSON')
    p_hist.add_argument('--store', help=store_help)
    p_hist.add_argument('--collection', help=coll_help)

    # serve
    p_serve = sub.add_parser('serve', help='Run a long-lived pre-write check daemon')
    p_serve.add_argument('memory_dir', help='Directory of markdown memory files')
    p_serve.add_argument('--socket', help='Unix socket path (default: <store>/vigil.sock)')
    p_serve.add_argument('--store', help=store_help)
    p_serve.add_argument('--collection', help=coll_help)

    # hook
    p_hook = sub.add_parser('hook', help='Install/run the git pre-commit gate')
    p_hook.add_argument('hook_action', choices=['install', 'run'],
                        help='install the pre-commit hook, or run the gate body')
    p_hook.add_argument('memory_dir', help='Directory of markdown memory files')
    p_hook.add_argument('--store', help=store_help)
    p_hook.add_argument('--collection', help=coll_help)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        'index': cmd_index,
        'scan': cmd_scan,
        'check': cmd_check,
        'health': cmd_health,
        'fix': cmd_fix,
        'history': cmd_history,
        'serve': cmd_serve,
        'hook': cmd_hook,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
