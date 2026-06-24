"""`vigil fix` — turn a scan into a RESOLUTION PLAN and optionally apply it.

Vigil historically only DETECTS. `fix` adds the act half: it runs a scan, builds
a per-category plan, prints it (dry-run by default), and under `--apply` performs
the strongest action Vigil ever takes — MOVING a file into an `archive/` subdir
of the memory dir. Vigil NEVER deletes; archive is reversible (the file is moved,
not removed) and only happens with `--apply`.

Per category:
  stale          : --apply archives the file (move to archive/).
  duplicates     : keep the higher-health / newer of the pair, archive the
                   loser — only when sim >= fix_duplicate_threshold (conservative).
  orphans        : list broken refs with file+line; suggest manual removal
                   (never edits file bodies).
  contradictions : surface suggested_keep (advisory); never auto-archive.
  provenance     : list the missing fields per file (cannot infer values).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Action:
    """A single planned action within the resolution plan."""
    category: str
    action: str            # 'archive', 'review', 'manual_edit', 'add_metadata'
    target: str            # file stem the action concerns
    reason: str
    applied: bool = False
    details: dict = field(default_factory=dict)


def build_plan(
    results: dict,
    memory_dir: Path,
    store_dir: Path,
    config,
    collection_name: str | None = None,
) -> list[Action]:
    """Build the resolution plan (list of Actions) from scan results.

    Pure planning — does not touch the filesystem. `config.fix_duplicate_threshold`
    gates which duplicates are confident enough to archive.
    """
    from .resolve import file_signals, choose_keep, load_health_data
    from .scanner import _get_access_data

    access_data = _get_access_data(store_dir, collection_name)
    health_data = load_health_data(store_dir, collection_name)

    actions: list[Action] = []

    # --- stale: archive ---
    for iss in results.get('stale', []):
        for f in iss.files:
            actions.append(Action(
                category='stale',
                action='archive',
                target=f,
                reason=iss.message,
                details={'staleness_score': iss.details.get('staleness_score'),
                         'severity': iss.severity},
            ))

    # --- duplicates: keep one, archive the loser (conservative) ---
    archived_dupe = set()
    for iss in results.get('duplicates', []):
        if len(iss.files) != 2:
            continue
        sim = iss.details.get('similarity', 0.0)
        a, b = iss.files
        sig_a = file_signals(memory_dir, a, access_data, health_data)
        sig_b = file_signals(memory_dir, b, access_data, health_data)
        keep, archive = choose_keep(sig_a, sig_b)
        confident = sim >= config.fix_duplicate_threshold
        if archive in archived_dupe:
            continue  # don't double-plan archiving the same loser
        actions.append(Action(
            category='duplicates',
            action='archive' if confident else 'review',
            target=archive,
            reason=(f'near-duplicate of {keep} (sim={sim:.2f}); '
                    + ('high-confidence -> archive loser'
                       if confident else
                       f'below fix threshold {config.fix_duplicate_threshold:.2f}'
                       ' -> review only')),
            details={'keep': keep, 'archive': archive, 'similarity': sim,
                     'confident': confident},
        ))
        if confident:
            archived_dupe.add(archive)

    # --- orphans: manual removal of broken refs (never auto-edit bodies) ---
    for iss in results.get('orphans', []):
        ref = iss.details.get('missing_path') or iss.details.get('missing_ref', '')
        line = iss.details.get('line', 0)
        target = iss.files[0] if iss.files else ''
        actions.append(Action(
            category='orphans',
            action='manual_edit',
            target=target,
            reason=f'broken ref {ref!r} at {target}.md:{line} — remove or fix manually',
            details={'ref': ref, 'line': line},
        ))

    # --- contradictions: advisory suggested_keep, never auto-act ---
    for iss in results.get('contradictions', []):
        keep = iss.details.get('suggested_keep', '')
        archive = iss.details.get('suggested_archive', '')
        actions.append(Action(
            category='contradictions',
            action='review',
            target=keep or (iss.files[0] if iss.files else ''),
            reason=(f'{iss.message}; suggested_keep={keep or "?"}'
                    f' (advisory — never auto-deleted)'),
            details={'files': iss.files, 'suggested_keep': keep,
                     'suggested_archive': archive,
                     'nli_score': iss.details.get('nli_score')},
        ))

    # --- provenance: list missing fields (cannot infer values) ---
    for iss in results.get('provenance', []):
        missing = iss.details.get('missing_fields', [])
        target = iss.files[0] if iss.files else ''
        actions.append(Action(
            category='provenance',
            action='add_metadata',
            target=target,
            reason=f'missing provenance fields: {", ".join(missing)} (add manually)',
            details={'missing_fields': missing},
        ))

    return actions


def archive_file(memory_dir: Path, stem: str, archive_dir_name: str) -> Path | None:
    """Move `<memory_dir>/<stem>.md` into `<memory_dir>/<archive_dir>/`.

    Creates the archive dir. On a name collision in the archive, a numeric
    suffix is added so nothing is overwritten. Returns the destination path, or
    None if the source file does not exist. NEVER deletes.
    """
    src = Path(memory_dir) / f'{stem}.md'
    if not src.exists():
        return None
    archive_dir = Path(memory_dir) / archive_dir_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / src.name
    i = 1
    while dest.exists():
        dest = archive_dir / f'{src.stem}.{i}.md'
        i += 1
    shutil.move(str(src), str(dest))
    return dest


def apply_plan(actions: list[Action], memory_dir: Path, config) -> list[Action]:
    """Apply the archive actions in the plan. Mutates Action.applied in place.

    Only 'archive' actions modify the filesystem (moving files into the archive
    dir). All other actions are advisory and left untouched.
    """
    for act in actions:
        if act.action != 'archive':
            continue
        dest = archive_file(memory_dir, act.target, config.archive_dir)
        if dest is not None:
            act.applied = True
            act.details['archived_to'] = str(dest)
    return actions


def plan_to_dict(actions: list[Action], applied: bool) -> dict:
    """Serialize the plan for `--json`."""
    by_cat: dict = {}
    for act in actions:
        by_cat.setdefault(act.category, []).append(asdict(act))
    n_archive = sum(1 for a in actions if a.action == 'archive')
    n_applied = sum(1 for a in actions if a.applied)
    return {
        'applied': applied,
        'summary': {
            'total_actions': len(actions),
            'archive_actions': n_archive,
            'archived': n_applied,
        },
        'actions': by_cat,
    }


def format_plan(actions: list[Action], applied: bool, archive_dir_name: str) -> str:
    """Render the resolution plan as readable text."""
    lines = [
        '=' * 60,
        '  VIGIL — Resolution Plan' + ('  [APPLIED]' if applied else '  [DRY RUN]'),
        '=' * 60,
        '',
    ]
    if not actions:
        lines.append('  Nothing to resolve — memory is healthy.')
        return '\n'.join(lines)

    n_archive = sum(1 for a in actions if a.action == 'archive')
    verb = 'archived' if applied else 'would archive'
    lines.append(f'  {len(actions)} planned action(s); {verb} {n_archive} file(s)'
                 f' into {archive_dir_name}/')
    lines.append('')

    order = ['stale', 'duplicates', 'contradictions', 'orphans', 'provenance']
    by_cat: dict = {}
    for act in actions:
        by_cat.setdefault(act.category, []).append(act)

    for cat in order:
        cat_actions = by_cat.get(cat, [])
        if not cat_actions:
            continue
        lines.append(f'--- {cat.upper()} ({len(cat_actions)}) ---')
        for act in cat_actions:
            tag = {
                'archive': '[ARCHIVE]',
                'review': '[REVIEW] ',
                'manual_edit': '[MANUAL] ',
                'add_metadata': '[META]   ',
            }.get(act.action, '[?]')
            status = ''
            if act.action == 'archive':
                status = ' (DONE)' if act.applied else ' (pending --apply)'
            lines.append(f'  {tag} {act.target}{status}')
            lines.append(f'           {act.reason}')
        lines.append('')

    if not applied and n_archive:
        lines.append('  Re-run with --apply to archive (files are MOVED, never deleted).')
    return '\n'.join(lines)


def run_fix(
    memory_dir: Path,
    store_dir: Path,
    config,
    collection_name: str | None = None,
    apply: bool = False,
    checks: list | None = None,
    now: datetime | None = None,
) -> tuple[list[Action], dict]:
    """Scan, build the plan, optionally apply. Returns (actions, scan_results)."""
    from .scanner import full_scan
    results = full_scan(
        memory_dir=memory_dir, store_dir=store_dir, checks=checks,
        collection_name=collection_name, config=config, now=now,
    )
    actions = build_plan(results, memory_dir, store_dir, config, collection_name)
    if apply:
        apply_plan(actions, memory_dir, config)
    return actions, results
