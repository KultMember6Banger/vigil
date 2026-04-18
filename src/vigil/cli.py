"""Vigil CLI — memory health monitor for AI agents.

Usage:
  vigil index  <memory_dir>              # Build/update search index
  vigil scan   <memory_dir>              # Full health scan
  vigil scan   <memory_dir> --check stale --check orphans  # Selective
  vigil check  <memory_dir> "new text"   # Pre-write contradiction check
  vigil check  <memory_dir> --file x.md  # Check file contents
  vigil health <memory_dir>              # Full scan + update health scores
"""

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_index(args):
    from .indexer import build_index

    memory_dir = Path(args.memory_dir)
    store_dir = Path(args.store) if args.store else None

    print(f'Vigil indexing {memory_dir}...')
    stats = build_index(
        memory_dir, store_dir=store_dir,
        full_rebuild=args.rebuild,
    )
    print(f'  Indexed: {stats["indexed"]} files'
          f' (skipped {stats["skipped"]} unchanged)')
    print(f'  Records: {stats["records"]} total in store')
    print(f'  Time:    {stats["elapsed"]}s')


def cmd_scan(args):
    from .scanner import full_scan, format_report
    from .indexer import default_store_dir

    memory_dir = Path(args.memory_dir)
    store_dir = Path(args.store) if args.store else default_store_dir(memory_dir)

    checks = args.checks or ['duplicates', 'isolated', 'orphans', 'stale', 'provenance', 'contradictions']

    print(f'Vigil scanning {memory_dir}...')
    t0 = time.time()

    step = 0
    total_steps = len(checks)

    results = {}
    if 'duplicates' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Duplicates...')
        from .scanner import find_duplicates
        results['duplicates'] = find_duplicates(store_dir)
    if 'isolated' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Isolated entries...')
        from .scanner import find_isolated
        results['isolated'] = find_isolated(store_dir)
    if 'orphans' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Orphans...')
        from .scanner import find_orphans
        results['orphans'] = find_orphans(memory_dir)
    if 'stale' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Staleness...')
        from .scanner import find_stale
        results['stale'] = find_stale(memory_dir, store_dir=store_dir)
    if 'provenance' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Provenance...')
        from .scanner import find_unprovenanced
        results['provenance'] = find_unprovenanced(memory_dir)
    if 'contradictions' in checks:
        step += 1
        print(f'  [{step}/{total_steps}] Contradictions (NLI model)...')
        from .scanner import find_contradictions
        results['contradictions'] = find_contradictions(store_dir)

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
    from .indexer import default_store_dir

    memory_dir = Path(args.memory_dir)
    store_dir = Path(args.store) if args.store else default_store_dir(memory_dir)

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
    issues = pre_write_check(text, store_dir=store_dir, source_file=source)
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
    from .indexer import default_store_dir

    memory_dir = Path(args.memory_dir)
    store_dir = Path(args.store) if args.store else default_store_dir(memory_dir)

    print('Vigil health scan + score update...')
    t0 = time.time()

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir)
    scores = compute_health_scores(results)
    n = update_health_scores(scores, store_dir)

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

    # index
    p_idx = sub.add_parser('index', help='Build/update search index')
    p_idx.add_argument('memory_dir', help='Directory of markdown memory files')
    p_idx.add_argument('--store', help='ChromaDB store path (default: <memory_dir>/.vigil/)')
    p_idx.add_argument('--rebuild', action='store_true', help='Full rebuild (delete existing index)')

    # scan
    p_scan = sub.add_parser('scan', help='Full health scan')
    p_scan.add_argument('memory_dir', help='Directory of markdown memory files')
    p_scan.add_argument('--check',
                        choices=['contradictions', 'duplicates', 'isolated', 'stale', 'orphans', 'provenance'],
                        action='append', dest='checks')
    p_scan.add_argument('--json', action='store_true', help='Output as JSON')
    p_scan.add_argument('--store', help='ChromaDB store path')

    # check (pre-write)
    p_check = sub.add_parser('check', help='Pre-write contradiction check')
    p_check.add_argument('memory_dir', help='Directory of markdown memory files')
    p_check.add_argument('text', nargs='?', help='Text to check')
    p_check.add_argument('--file', help='Read text from file instead')
    p_check.add_argument('--source', default='', help='Source file stem to exclude')
    p_check.add_argument('--store', help='ChromaDB store path')

    # health
    p_health = sub.add_parser('health', help='Full scan + update health scores')
    p_health.add_argument('memory_dir', help='Directory of markdown memory files')
    p_health.add_argument('--store', help='ChromaDB store path')

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
