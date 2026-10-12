#!/usr/bin/env python3
"""Unattended registration of renders and revisions of the shared target training set (DGX render agent).

One pass (``--loop`` repeats it every ``--interval`` seconds; the state lives in files, so the unit is resumable):

1. Render jobs (``runs/natlang-target-set-20261011/jobs.json``: id, render_dir, source_corpora, supersedes, notes):
   a finished render (``receipt.json``) whose id is not on origin/main is registered and its manifest published through
   register_corpora.py (private worktree off origin/main, retried with back-off when the registry is busy). Renders
   that other agents register themselves (static curriculum batches, Clef pools, SWE twins, S1 calls) are never
   registered here: they join the set through the registry.
2. Membership (target_set_manifest.resolve over origin/main's registry and the living spec). When it differs from the
   last revision's and the last union build is older than ``--min-gap`` seconds (or ``--force``), revision N+1:
   - the union is built by build_native_turn_union.py as its own ledger unit (low priority, resumable) into
     ``runs/natlang-target-union-20261011-r<N>`` (one cross-deduplicated, decontaminated, twin-aligned file per
     rendering and tokenizer; ``.zst`` members are streamed through the host's zstd, mounted into the container);
   - ``r<N>.json`` (members, files, skipped with reasons, pending) is written to ``runs/natlang-target-set-20261011``;
   - ``natlang-target-set-20261011-r<N>`` and ``natlang-target-union-20261011-r<N>`` are registered together;
   - the trainer is told the new ids and union files via coord (it resumes onto newer revisions);
   - union files older than the previous revision are deleted (derived, reproducible from their receipt); their
     registry entries stay and say so.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import root, resolve  # noqa: E402
import target_set_manifest as tsm  # noqa: E402
from register_corpora import register, registered_ids  # noqa: E402
import register_native_turn_render as rntr  # noqa: E402

SET_DIR = tsm.SET_DIR
JOBS = SET_DIR / 'jobs.json'
STATE = SET_DIR / 'watcher-state.json'
UNION_BASE = 'natlang-target-union-20261011'
STAGING = resolve('data_nvme', 'union-staging')
LFM = ROOT / 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f'
MELLUM = resolve('models', 'mellum21-12b-a2.5b-thinking')


def log(**kw):
    print(json.dumps({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), **kw}), flush=True)


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def register_jobs(known: set[str]) -> list[str]:
    done = []
    for job in load(JOBS, []):
        out = ROOT / job['render_dir']
        if job['id'] in known or not (out / 'receipt.json').exists():
            continue
        argv = ['--id', job['id'], '--render-dir', str(out), '--notes', job.get('notes', '')]
        for src in job['source_corpora']:
            argv += ['--source-corpus', src]
        for sup in job.get('supersedes', []):
            argv += ['--supersedes', sup]
        entry = rntr.make_entry(rntr.parse(argv))
        register([entry], f"Register {job['id']}")
        log(registered=job['id'])
        done.append(job['id'])
    return done


def staged(rel: str, rev: int) -> str:
    """The path the union container reads. .zst members are streamed by the builder (the host's zstd is mounted into
    the container), never decompressed to staging: the union is then the only plain copy of a compressed member
    (2026-10-12; the SWE bulk twins are ~110 GB plain, ~2.5 GB compressed)."""
    return str(ROOT / rel)


def ledger_knows(unit: str) -> bool:
    status = subprocess.run([sys.executable, str(ROOT / 'scripts/memory_ledger.py'), 'status'], cwd=ROOT,
                            capture_output=True, text=True)
    try:
        d = json.loads(status.stdout)
    except ValueError:
        return True  # unknown: do not launch a duplicate
    names = set(d.get('claims', {})) | {w['unit'] for w in d.get('waiting', [])}
    return f'{unit}.service' in names


def build_union(members: list[dict], rev: int, launches: int = 0) -> Path | str | None:
    """The union directory once built; None while its ledger unit is queued or running; 'launched' when started here."""
    out = ROOT / 'runs' / f'{UNION_BASE}-r{rev}'
    if (out / 'receipt.json').exists():
        shutil.rmtree(STAGING / f'r{rev}', ignore_errors=True)
        return out
    unit = f'union-target-r{rev}'
    if ledger_knows(unit) or subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{unit}.service']).returncode == 0:
        return None
    if launches >= 3:
        raise RuntimeError(f'union r{rev} failed {launches} times; see {SET_DIR}/union-r{rev}.launch.log and '
                           f'journalctl --user -u {unit}; delete "building" in {STATE} to retry')
    args = []
    for m in members:
        for rendering, toks in m['files'].items():
            for tok, rel in toks.items():
                args += ['--member', f"{m['id']}:{rendering}:{tok}={staged(rel, rev)}"]
    out.mkdir(parents=True, exist_ok=True)
    zstd = shutil.which('zstd') or '/usr/bin/zstd'
    mounts = ['-v', f'{os.path.realpath(zstd)}:/usr/local/bin/zstd:ro',
              '-v', f'{ROOT}:{ROOT}:ro', '-v', f'{root("data_hdd")}:{root("data_hdd")}:ro',
              '-v', f'{root("data_nvme")}:{root("data_nvme")}',
              '-v', f'{out}:{out}']
    cmd = [sys.executable, str(ROOT / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', '10',
           '--priority', 'low', '--resumable', '--wait', '86400', '--workdir', str(ROOT), '--',
           'docker', 'run', '--rm', '--name', unit, '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
           '--memory', '12g', '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
           *mounts, '-w', str(ROOT), '--entrypoint', 'python3', 'natlang-neuralese:26.09',
           'scripts/neuralese_data/build_native_turn_union.py', '--out-dir', str(out),
           '--tokenizer', f'lfm25={LFM}', '--tokenizer', f'mellum21={MELLUM}', *args]
    log_path = SET_DIR / f'union-r{rev}.launch.log'
    subprocess.Popen(cmd, cwd=ROOT, stdout=open(log_path, 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    log(union_launched=rev, unit=unit, members=len(members))
    return 'launched'


def registry_sources(members: list[dict]) -> list[str]:
    return sorted({i for m in members for i in (m.get('registry_ids') or [m['id']])})


def finish_revision(state: dict, build: dict, union: Path) -> None:
    rev, members, skipped, spec, sig = build['rev'], build['members'], build['skipped'], build['spec'], build['signature']
    union_receipt = json.loads((union / 'receipt.json').read_text())
    totals = {r: {t: {'documents': s['documents'], 'by_split': s['by_split']} for t, s in v['by_tokenizer'].items()}
              for r, v in union_receipt['renderings'].items()}
    set_id, union_id = f'{tsm.BASE_ID}-r{rev}', f'{UNION_BASE}-r{rev}'
    snapshot = {'id': set_id, 'revision': rev, 'signature': sig, 'union': union_id,
                'members': members, 'skipped': skipped, 'pending': spec.get('pending', []),
                'exclude': spec.get('exclude', []), 'totals_after_union': totals}
    save(SET_DIR / f'r{rev}.json', snapshot)
    argv = ['--id', union_id, '--render-dir', str(union), '--kind', 'union',
            '--notes', f'Union of target-set revision {set_id}: one file per rendering and tokenizer, cross-member '
            'exact deduplication, held stays held, source-group-disjoint decontamination across all members, tokenizer '
            'twins aligned by document id (build_native_turn_union.py). Written by target_set_watcher.py.']
    for src in registry_sources(members):
        argv += ['--source-corpus', src]
    union_entry = rntr.make_entry(rntr.parse(argv))
    prev = state.get('revision')
    set_entry = {
        'id': set_id, 'owner': 'dgx', 'path': os.path.relpath(SET_DIR, ROOT), 'include': [f'r{rev}.json'],
        'source_class': 'shared-natlang-target-training-set-manifest',
        'admission': 'manifest of training-admitted member renders (owner 2026-10-11: expand the target set by '
                     'everything found that can be integrated); exclusions only protected eval sets, held splits and '
                     'defect rows under repair',
        'training_admission': True,
        'training_admission_reason': f'lists training-admitted members; the training input is {union_id}',
        'members': [m['id'] for m in members], 'skipped': skipped, 'union': union_id, 'totals_after_union': totals,
        'notes': 'Written by scripts/neuralese_data/target_set_watcher.py from the living spec '
                 'runs/natlang-target-set-20261011/spec.json and the registry on origin/main.',
    }
    if prev:
        set_entry['supersedes'] = [f'{tsm.BASE_ID}-r{prev}']
        if (ROOT / 'runs' / f'{UNION_BASE}-r{prev}').exists():
            union_entry['supersedes'] = [f'{UNION_BASE}-r{prev}']
    register([union_entry, set_entry], f'Target set revision {set_id} and its union')
    log(revision=rev, union=union_id, totals=totals)
    rel = os.path.relpath(union, ROOT)
    files = {r: {t: f'{rel}/{r}/{t}/text-data.jsonl' for t in v['by_tokenizer']}
             for r, v in union_receipt['renderings'].items()}
    shas = {r: {t: s['text_data_sha256'] for t, s in v['by_tokenizer'].items()}
            for r, v in union_receipt['renderings'].items()}
    subprocess.run([sys.executable, str(ROOT / 'scripts/coord.py'), '--as', 'dgx-render', 'send', '--to', 'dgx',
                    '--subject', f'Target set {set_id}: union {union_id} registered (training input for both lines)',
                    '-m', json.dumps({'union': union_id, 'set': set_id, 'files': files, 'sha256': shas,
                                      'totals': totals, 'members': [m['id'] for m in members]}, indent=1)],
                   cwd=ROOT, check=False)
    kept = state.get('union_revisions', [])
    for old in kept[:-1]:  # keep the previous union for a trainer still resuming on it; older ones are reproducible
        shutil.rmtree(ROOT / 'runs' / f'{UNION_BASE}-r{old}', ignore_errors=True)
        log(deleted_union=old)
    state.update(revision=rev, signature=sig, union_revisions=kept[-1:] + [rev])
    state.pop('building', None)
    save(STATE, state)


def one_pass(min_gap: float, force: bool) -> None:
    register_jobs(registered_ids())
    spec = load(tsm.SPEC, {})
    state = load(STATE, {'revision': spec.get('revision', 1)})
    build = state.get('building')
    if build:
        union = build_union(build['members'], build['rev'], build.get('launches', 0))
        if union == 'launched':
            build['launches'] = build.get('launches', 0) + 1
            save(STATE, state)
        elif union is not None:
            finish_revision(state, build, union)
        return
    members, skipped = tsm.resolve(tsm.origin_registry(), spec)
    sig = tsm.signature(members)
    if sig == state.get('signature'):
        return
    if not force and time.time() - state.get('last_build', 0) < min_gap:
        log(waiting_for_gap=True, members=len(members))
        return
    state['building'] = {'rev': state.get('revision', 1) + 1, 'signature': sig, 'members': members,
                         'skipped': skipped, 'spec': spec}
    state['last_build'] = time.time()
    if build_union(members, state['building']['rev']) == 'launched':
        state['building']['launches'] = 1
        save(STATE, state)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loop', action='store_true')
    p.add_argument('--interval', type=float, default=600)
    p.add_argument('--min-gap', type=float, default=3 * 3600, help='seconds between union rebuilds')
    p.add_argument('--force', action='store_true', help='build a revision now if membership changed')
    a = p.parse_args()
    while True:
        try:
            one_pass(a.min_gap, a.force)
        except Exception as exc:  # noqa: BLE001 - a failed pass is logged and retried next interval
            log(error=repr(exc)[:2000])
        a.force = False
        if not a.loop:
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
