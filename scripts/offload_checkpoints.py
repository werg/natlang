#!/usr/bin/env python3
"""One-shot checkpoint offload for already-written checkpoints of active runs (owner 2026-10-13).

The same operation the trainers now run in the background after every write (training/neuralese/natlang_neuralese/
train/checkpoint_offload.py; plans/STORAGE_POLICY.md section 2a), applied to the files already on the NVMe:

- non-latest checkpoints (``best-*.pt``, ``final-*.pt``, ``checkpoint-*.pt``, ``step-*.pt``): verified copy to the
  mirror ``<data_hdd>/natlang-active-lineage/<path relative to runs>``, then an atomic symlink in place;
- the resumable slot ``checkpoint.pt``: verified mirror copy only (it stays on the NVMe), unless ``--no-latest``.

Skipped: symlinks, files under 1 GiB, a ``.keep-on-nvme`` marker, a file with another copier's ``<mirror>.part``,
an archive-role file some process holds open. Dry run by default: lists exact paths, roles, sizes and actions.

    python3 scripts/offload_checkpoints.py                     # dry run over the default active runs
    python3 scripts/offload_checkpoints.py --apply [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.train import checkpoint_offload as co  # noqa: E402

GIB = 2**30
DEFAULT_RUNS = ('neuralese-direct-mellum-20261011', 'luna-neuralese-direct-lfm-base-20261011-v8*')


def _archive_module():
    spec = importlib.util.spec_from_file_location('archive_to_external', REPO / 'scripts' / 'archive_to_external.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(run_dirs, settings, *, latest=True, held=frozenset()):
    rows = []
    for run in run_dirs:
        for dirpath, _, _ in os.walk(run, followlinks=False):
            for path, role in co.sweep_targets(Path(dirpath), settings, own=latest):
                mirror = co.mirror_path(path, settings)
                st = path.stat()
                row = {'path': str(path), 'role': role, 'gb': round(st.st_size / GIB, 2), 'bytes': st.st_size,
                       'nlink': st.st_nlink, 'mirror': str(mirror) if mirror else None}
                reason = co.mirror_unavailable(path, settings)
                if reason is None and co.foreign_partial(mirror):
                    reason = f'another copy in progress: {co.foreign_partial(mirror)}'
                if reason is None and role == 'archive' and str(path) in held:
                    reason = 'open by a process'
                if reason:
                    row['action'] = 'skip: ' + reason
                elif co.mirror_current(path, mirror):
                    row['action'] = 'link (mirror current)' if role == 'archive' else 'none (mirror current)'
                else:
                    row['action'] = 'copy, verify, link' if role == 'archive' else 'copy, verify (stays on NVMe)'
                rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', type=Path, help='run directories (default: the active Mellum v4 lineage and '
                                                       "Pop's mirrored LFM v8 runs)")
    p.add_argument('--apply', action='store_true')
    p.add_argument('--no-latest', action='store_true', help='do not mirror the resumable slots')
    a = p.parse_args(argv)
    runs = a.runs or sorted({d for pattern in DEFAULT_RUNS for d in (REPO / 'runs').glob(pattern) if d.is_dir()})
    settings = co.OffloadSettings.from_env()
    rows = plan(runs, settings, latest=not a.no_latest, held=_archive_module().open_paths())
    nvme = REPO / 'runs'
    summary = {'apply': a.apply, 'runs': [str(r) for r in runs], 'mirror_root': settings.mirror_root,
               'nvme_free_gb': round(shutil.disk_usage(nvme).free / GIB, 1), 'files': rows}
    if a.apply:
        lock = open(Path.home() / '.local' / 'state' / 'natlang' / 'offload-checkpoints.lock', 'w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        co._lower_priority(os.getpid())
        done, failed = [], []
        for row in rows:
            if row['action'].startswith(('skip', 'none')):
                continue
            try:
                event = co.offload_one(row['path'], row['role'], settings)
                done.append(event)
                print(json.dumps(event), flush=True)
            except Exception as error:
                failed.append({'path': row['path'], 'error': repr(error)[:300]})
        summary.update(done=done, failed=failed, nvme_free_after_gb=round(shutil.disk_usage(nvme).free / GIB, 1),
                       linked_gb=round(sum(r['bytes'] for r in rows for e in done
                                           if e.get('linked') and e['path'] == r['path']) / GIB, 1))
    print(json.dumps(summary, indent=1))
    return 1 if a.apply and summary.get('failed') else 0


if __name__ == '__main__':
    sys.exit(main())
