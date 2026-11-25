#!/usr/bin/env python3
"""Standing ingestion of a live teacher campaign (``vibestudio_campaign.py``) into one growing sft_sources WORK.

The campaign writes numbered wave directories (and ``seed-*``, recovery ones) under its root, each committed by its
``exported.json``: written last, after ``receipt.json`` and the wave's trajectory export (``export_trajectories``). A
directory without that marker is still being written and is not read. For every committed directory not yet ingested
this takes the receipt's verified manifests and, per episode (``live_teacher_sources.load_episode``, the bound call,
validator and effect artifacts checked):

  1. held separation: ``sft_sources.held_excluded_ids`` (``held_eval.exclusion_ids`` of held-v3 and the held-v2 it
     excludes, plus source groups and source ids of the held closure) with the campaign catalog's own validation and
     test open goals added (``live_holdout_registry``), dropped as ``held``;
  2. the goal must match the catalog's training registration (``require_registered_open_goal``: id, source groups,
     goal hash), else ``unregistered_open_goal``;
  3. an episode key already ingested (``duplicate_episode``) or a completed goal already ingested (``repeated_open_goal``). Independently checked partial decisions
     use episode identity and never reserve a whole goal is dropped.

A scan's kept manifests are extracted together by ``sft_sources.extract --live-run`` into
``WORK/episodes-<first directory with episodes>.jsonl.zst`` (extract registers the WORK with the manifests as inputs;
the build and filled_reasoning read every ``episodes*.jsonl.zst``, each key once), then one line per directory is
appended to ``WORK/ingest.jsonl`` (the ingested marker: a rerun skips it; a scan cut off before its lines is extracted
again). The catalog is copied once to ``WORK/live-holdout-registry.jsonl``
(the build's ``--live-holdout-registry``) and remains immutable. Later catalogs must preserve its exact task/held identity; inventoried execution/review metadata may change with a source-hashed compatibility receipt. Original campaign data is never modified. After durable ingestion, unreferenced native diagnostic pack
infrastructure excluded from all registered snapshots may be retired, with a small retirement receipt;
original trajectories, IR, source and verification evidence remain unchanged.

    live_campaign_ingest.py once --campaign ROOT --work WORK [--held-manifest HELD]
    live_campaign_ingest.py poll --campaign ROOT --work WORK --every 1800
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common import data_registry  # noqa: E402

SCHEMA = 'natlang.live-campaign-ingest/1'
REGISTRY = 'live-holdout-registry.jsonl'


def committed(campaign):
    """The campaign's committed directories (``exported.json`` and ``receipt.json`` present), in name order."""
    return [d for d in sorted(Path(campaign).iterdir())
            if d.is_dir() and (d / 'exported.json').is_file() and (d / 'receipt.json').is_file()]


def verified_manifests(directory):
    """The verified manifests a committed directory's receipt names (paths under ``capture/``), checked against the
    count its ``exported.json`` committed."""
    exported = json.loads((directory / 'exported.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    from live_teacher_sources import has_training_manifest
    manifests = [directory / 'capture' / r['manifest'] for _, r in sorted(receipt.items()) if has_training_manifest(r)]
    if len(manifests) != exported.get('verified'):
        raise ValueError(f'{directory}: receipt lists {len(manifests)} verified, exported.json {exported.get("verified")}')
    return manifests, exported.get('dataset')


def ingested(work):
    path = Path(work) / 'ingest.jsonl'
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def holdout_registry(campaign, work):
    """Immutable WORK registration; explicitly inventoried execution metadata may evolve."""
    from live_catalog_identity import compatible_catalogs
    source, copy = Path(campaign) / 'catalog.jsonl', Path(work) / REGISTRY
    if not copy.exists():
        temporary = copy.with_suffix('.partial')
        shutil.copyfile(source, temporary)
        os.replace(temporary, copy)
    try:
        receipt = compatible_catalogs(copy, source)
    except (ValueError, TypeError, KeyError) as error:
        raise SystemExit(f'{source} changed since {copy} was taken: {error}; a new identity needs a new WORK') from error
    receipt_path = Path(work) / 'catalog-compatibility.json'
    temporary = receipt_path.with_suffix('.partial')
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, receipt_path)
    return copy


def select(manifests, held_manifest, held, training, seen_keys, seen_goals, load=None, dependencies=None):
    """(kept manifests, their records, drop counts) of one directory's manifests (the module's steps 1-3)."""
    from sft_sources import held_excluded_ids, open_goal_key, require_registered_open_goal
    if load is None:
        from live_teacher_sources import load_episode as load
    kept, records, dropped = [], [], collections.Counter()
    for path in manifests:
        from live_teacher_sources import observed_artifacts
        with observed_artifacts() as artifact_paths:
            episode = load(path)
        if dependencies is not None:
            dependencies.update(artifact_paths)
        task = episode['task']
        goal = (task.get('interaction') or {}).get('goal')
        if held_excluded_ids(held_manifest, {episode['task_id']: task}, held):
            dropped['held'] += 1
            continue
        try:
            require_registered_open_goal({'task_id': episode['task_id'], 'source_groups': task['source_groups'],
                                          'open_goal': goal}, training)
        except (ValueError, TypeError):
            dropped['unregistered_open_goal'] += 1
            continue
        if episode['key'] in seen_keys:
            dropped['duplicate_episode'] += 1
            continue
        completed = (episode.get('trajectory_ir') or {}).get('outcome', {}).get('task_completed', True)
        supervised = bool(episode['turns'])
        if completed and supervised and open_goal_key(goal) in seen_goals:
            dropped['repeated_open_goal'] += 1
            continue
        seen_keys.add(episode['key'])
        if completed and supervised:
            seen_goals.add(open_goal_key(goal))
        kept.append(path)
        records.append({'key': episode['key'], 'task_id': episode['task_id'], 'goal': open_goal_key(goal),
                        'decisions': len(episode['turns']), 'task_completed': completed})
    return kept, records, dropped


def ingest_once(campaign, work, held_manifest=None, extract=None, *, registry_repo=None, registry_ref=None):
    """Ingest every committed, not yet ingested directory; returns the new ingest records."""
    from sft_sources import held_closure, live_holdout_registry
    if extract is None:
        from sft_sources import extract
    registry_repo = (data_registry.assert_registry_writable(registry_repo) if registry_repo is not None
                     else data_registry.registry_owner_repo())
    registry_ref = registry_ref if registry_ref is not None else data_registry.registry_owner_ref(registry_repo)
    campaign, work = Path(campaign).resolve(), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    held_manifest = held_manifest or data_registry.role_path('held-eval')
    with (work / '.ingest.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = holdout_registry(campaign, work)
        held = held_closure(held_manifest)
        training = live_holdout_registry(registry, held)       # adds the catalog's held open goals to ``held``
        earlier = ingested(work)
        done = {r['directory'] for r in earlier}
        seen_keys = {e['key'] for r in earlier for e in r['episodes']}
        seen_goals = {e['goal'] for r in earlier for e in r['episodes'] if e.get('task_completed', True) and e.get('decisions', 1) > 0}
        batch = []
        observed_by_directory = {}
        for directory in committed(campaign):
            if directory.name in done:
                continue
            manifests, dataset = verified_manifests(directory)
            dependencies = set()
            kept, records, dropped = select(manifests, held_manifest, held, training, seen_keys, seen_goals,
                                            dependencies=dependencies)
            observed_by_directory[directory.name] = sorted(dependencies)
            batch.append((directory, dataset, manifests, kept, records, dropped))
        # one extraction per scan (each registry lookup of a campaign path indexes every dataset directory: minutes
        # under HDD load), named after its first directory with episodes; a scan cut off before its lines repeats it
        kept_all = [str(m) for _, _, _, kept, _, _ in batch for m in kept]
        name = next((f'episodes-{d.name}' for d, _, _, kept, _, _ in batch if kept), None)
        if kept_all:
            extract(types.SimpleNamespace(work=str(work), rl=[], teacher=[], live_run=kept_all, live_capture_root=[],
                                          live_trajectory_out=None, out_name=name, october_teacher=[],
                                          october_model=[], held_manifest=held_manifest,
                                          registry_repo=registry_repo, registry_ref=registry_ref))
        new = []
        with (work / 'ingest.jsonl').open('a') as handle:
            for directory, dataset, manifests, kept, records, dropped in batch:
                record = {'schema': SCHEMA, 'directory': directory.name, 'dataset': dataset,
                          'verified': len(manifests), 'kept': len(kept), 'dropped': dict(dropped), 'episodes': records,
                          'artifact_paths': observed_by_directory[directory.name],
                          'extraction': f'{name}.jsonl.zst' if kept else None, 'time': time.time()}
                handle.write(json.dumps(record) + '\n')
                print(json.dumps({k: record[k] for k in ('directory', 'verified', 'kept', 'dropped')}), flush=True)
                new.append(record)
            handle.flush()
            os.fsync(handle.fileno())
    # The journal and extraction are durable before any generated-state retirement.
    from retained_state_archive import retire_ingested_native_state
    for record in [*earlier, *new]:
        try:
            retire_ingested_native_state(campaign / record['directory'], record, work=work,
                                         registry_repo=registry_repo, registry_ref=registry_ref)
        except (OSError, ValueError) as error:
            # Ingestion already committed. Keep recovery state and retry later;
            # housekeeping must never hold training or discard a trajectory.
            print(f"Native state retained for {record['directory']}: {error}", file=sys.stderr)
    return new


def summary(work):
    records = ingested(work)
    dropped = collections.Counter()
    for r in records:
        dropped.update(r['dropped'])
    return {'directories': len(records), 'verified': sum(r['verified'] for r in records),
            'episodes': sum(r['kept'] for r in records), 'dropped': dict(dropped)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('once', 'poll'):
        s = sub.add_parser(name)
        s.add_argument('--campaign', required=True, help='the campaign root (status.json, catalog.jsonl, wave dirs)')
        s.add_argument('--work', required=True, help='the sft_sources WORK this keeps growing')
        s.add_argument('--held-manifest', default=None, help='default: the registry role held-eval (held-v3)')
        s.add_argument('--registry-repo', default=None, help='mutable dataset registry owner; code and held selection stay pinned')
        s.add_argument('--registry-ref', default=None, help='published Git snapshot; default current owner publication')
        if name == 'poll':
            s.add_argument('--every', type=float, default=1800., help='seconds between scans')
    args = p.parse_args()
    while True:
        try:
            ingest_once(args.campaign, args.work, args.held_manifest, registry_repo=args.registry_repo,
                        registry_ref=args.registry_ref)
        except Exception as error:
            if args.command == 'once':
                raise
            print(json.dumps({'event': 'ingest failed', 'error': repr(error)[:600], 'time': time.time()}), flush=True)
        print(json.dumps({'event': 'scanned', **summary(args.work), 'time': time.time()}), flush=True)
        if args.command == 'once':
            return
        time.sleep(args.every)


if __name__ == '__main__':
    main()
