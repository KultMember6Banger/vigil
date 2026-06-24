"""Vigil CLI — memory health monitor for AI agents.

Usage:
  vigil index  <memory_dir>              # Build/update search index
  vigil scan   <memory_dir>              # Full health scan
  vigil scan   <memory_dir> --check stale --check orphans  # Selective
  vigil check  <memory_dir> "new text"   # Pre-write contradiction check
  vigil check  <memory_dir> --file x.md  # Check file contents
  vigil health <memory_dir>              # Full scan + update health scores

Store dir resolves to --store, else $MEMORY_STORE, else <memory_dir>/.vigil/.
Collection resolves to --collection, else $MEMORY_COLLECTION, else 'agent_memory'.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _resolve_store(args, memory_dir):
    """--store override, else env-aware default (MEMORY_STORE / .vigil)."""
    from .indexer import default_store_dir
    return Path(args.store) if args.store else default_store_dir(memory_dir)


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
    from .scanner import (
        format_report, find_duplicates, find_isolated, find_orphans,
        find_stale, find_unprovenanced, find_contradictions,
        _get_all_records, _build_sim_matrix,
    )

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    coll = args.collection

    checks = args.checks or ['duplicates', 'isolated', 'orphans', 'stale', 'provenance', 'contradictions']

    print(f'Vigil scanning {memory_dir}...')
    t0 = time.time()

    step = 0
    total_steps = len(checks)

    # Read embeddings + build the similarity matrix once, shared across the
    # chroma-backed checks (avoids 3x redundant Chroma reads / matrix rebuilds).
    records = None
    sim_matrix = None
    if any(c in checks for c in ('duplicates', 'isolated', 'contradictions')):
        records = _get_all_records(store_dir, coll)
        if records and records.get('ids'):
            sim_matrix = _build_sim_matrix(records['embeddings'])

    results = {}
    if 'duplicates' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Duplicates...')
        results['duplicates'] = find_duplicates(
            store_dir, collection_name=coll, records=records, sim_matrix=sim_matrix)
    if 'isolated' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Isolated entries...')
        results['isolated'] = find_isolated(
            store_dir, collection_name=coll, records=records, sim_matrix=sim_matrix)
    if 'orphans' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Orphans...')
        results['orphans'] = find_orphans(memory_dir)
    if 'stale' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Staleness...')
        results['stale'] = find_stale(memory_dir, store_dir=store_dir, collection_name=coll)
    if 'provenance' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Provenance...')
        results['provenance'] = find_unprovenanced(memory_dir)
    if 'contradictions' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Contradictions (NLI model)...')
        results['contradictions'] = find_contradictions(
            store_dir, collection_name=coll, records=records, sim_matrix=sim_matrix)

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
    issues = pre_write_check(text, store_dir=store_dir, source_file=source,
                             collection_name=args.collection)
    elapsed = time.time() - t0

    if not issues:
        print(f'  CLEAR — no contradictions found ({elapsed:.1f}s)')
    else:
        print(f'\n  {len(issues)} potential conflicts found ({elapsed:.1f}s):')
        print()
        for issue in issues:
            icon = '!!!' if issue.severity == 'CRITICAL' else ' ! '
            print(f'  [{icon}] {issue.message}')
            print(f'         new:      {issue.details["new_text"][:120]}')
            print(f'         existing: {issue.details["existing_text"][:120]}')
            print()


def cmd_health(args):
    from .scanner import full_scan, compute_health_scores, update_health_scores

    memory_dir = Path(args.memory_dir)
    store_dir = _resolve_store(args, memory_dir)
    coll = args.collection

    print('Vigil health scan + score update...')
    t0 = time.time()

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir, collection_name=coll)
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

    # check (pre-write)
    p_check = sub.add_parser('check', help='Pre-write contradiction check')
    p_check.add_argument('memory_dir', help='Directory of markdown memory files')
    p_check.add_argument('text', nargs='?', help='Text to check')
    p_check.add_argument('--file', help='Read text from file instead')
    p_check.add_argument('--source', default='', help='Source file stem to exclude')
    p_check.add_argument('--store', help=store_help)
    p_check.add_argument('--collection', help=coll_help)

    # health
    p_health = sub.add_parser('health', help='Full scan + update health scores')
    p_health.add_argument('memory_dir', help='Directory of markdown memory files')
    p_health.add_argument('--store', help=store_help)
    p_health.add_argument('--collection', help=coll_help)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'check':
        cmd_check(args)
    elif args.command == 'health':
        cmd_health(args)


if __name__ == '__main__':
    main()
