#!/usr/bin/env python3
"""Hand a complete support-only headroom screen to a pinned skill collection queue."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading

from run_skill_authoring_queue import run_queue, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n') if line.strip()]


def validate_handoff(args):
    if digest(args.screen_input) != args.input_sha256:
        raise ValueError('screen input changed')
    if digest(args.runtime / 'frozen-runtime.json') != args.runtime_sha256:
        raise ValueError('runtime identity changed')
    episodes = [row for row in rows(args.screen_input) if row['split'] == 'train']
    if not episodes or len({row['id'] for row in episodes}) != len(episodes):
        raise ValueError('nonempty unique train packet required')
    all_ids = {row['id'] for row in episodes}
    batch_path = getattr(args, 'episode_ids', None)
    batch_hash = None
    if batch_path:
        batch_hash = digest(batch_path)
        if batch_hash != args.episode_ids_sha256:
            raise ValueError('predeclared episode batch changed')
        batch_ids = json.loads(batch_path.read_text())
        if not isinstance(batch_ids, list) or not batch_ids or any(not isinstance(x, str) for x in batch_ids) or len(set(batch_ids)) != len(batch_ids) or not set(batch_ids) <= all_ids:
            raise ValueError('invalid predeclared episode batch')
        episodes = [row for row in episodes if row['id'] in set(batch_ids)]
    screen_bytes = args.screen_out.read_bytes()
    all_screens = [json.loads(line) for line in screen_bytes.split(b'\n') if line.strip()]
    if len({row['episode'] for row in all_screens}) != len(all_screens) or any(row['episode'] not in all_ids for row in all_screens):
        raise ValueError('duplicate or foreign screen row')
    selected_ids = {row['id'] for row in episodes}
    screens = [row for row in all_screens if row['episode'] in selected_ids]
    if len(screens) != len(episodes) or len({row['episode'] for row in screens}) != len(screens):
        raise ValueError('screen must account for every train episode exactly once')
    by_id = {row['episode']: row for row in screens}
    if set(by_id) != {row['id'] for row in episodes}:
        raise ValueError('screen episode set differs')
    identities = {row['screen'] for row in screens}
    if len(identities) != 1:
        raise ValueError('mixed screen identities')
    executor_id = f"{getattr(args, 'executor_endpoint', None) or args.endpoint}:{getattr(args, 'executor_model', None) or args.model}"
    expected = []
    held = []
    for episode in episodes:
        screen = by_id[episode['id']]
        if screen.get('input_sha256') != args.input_sha256 or screen.get('executor') != executor_id:
            raise ValueError('screen input/executor identity differs')
        if screen.get('schema') != 'natlang.episode-headroom/2':
            raise ValueError('screen contract differs')
        quality = screen.get('support_quality')
        if screen.get('error') or screen.get('skipped'):
            held.append({'episode': episode['id'], 'reason': screen.get('error') or screen.get('skipped')})
            continue
        if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not 0 <= quality <= 1:
            raise ValueError('invalid support quality')
        if screen.get('cases') != len(episode['support']['cases']):
            raise ValueError('support screen case count differs')
        if args.low <= quality <= args.high:
            expected.append({**episode, 'provenance': {**episode.get('provenance', {}), 'headroom': {
                'executor': screen['executor'], 'screen': screen['screen'],
                'support_quality': quality, 'band': [args.low, args.high]}}})
        else:
            held.append({'episode': episode['id'], 'reason': 'support_quality_outside_declared_band', 'quality': quality})
    if batch_path:
        body = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in expected).encode()
        if args.kept.exists() and args.kept.read_bytes() != body:
            raise ValueError('immutable batch kept packet changed')
        if not args.kept.exists():
            args.kept.parent.mkdir(parents=True, exist_ok=True)
            with args.kept.open('xb') as stream:
                stream.write(body)
    elif rows(args.kept) != expected:
        raise ValueError('kept packet differs from independently selected source episodes')
    return {'schema': 'natlang.screened-skill-handoff/1', 'input_sha256': args.input_sha256,
        'runtime_manifest_sha256': args.runtime_sha256,
        'screened_batch_ids_sha256': batch_hash,
        'screen_rows': screens,
        'screen_sha256': hashlib.sha256(json.dumps(screens, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'kept_sha256': digest(args.kept), 'screen_identity': next(iter(identities)),
        'screened': len(episodes), 'selected': len(expected), 'held': held,
        'band': [args.low, args.high], 'selection': 'Support-only; no query/transfer outcome selection',
        'publication': 'Collection candidates only; offline replay and admission review still required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['screen-input', 'screen-out', 'kept', 'runtime', 'node', 'out']:
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ['input-sha256', 'runtime-sha256', 'screen-service']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--episode-ids', type=Path, help='Hash-pinned batch of train IDs; partition independently of outcome values')
    parser.add_argument('--episode-ids-sha256')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8082')
    parser.add_argument('--model', default='nvidia/Qwen3.6-35B-A3B-NVFP4')
    parser.add_argument('--executor-endpoint', help='must match the support-screen executor endpoint')
    parser.add_argument('--executor-model', help='must match the support-screen executor model')
    parser.add_argument('--low', type=float, default=0)
    parser.add_argument('--high', type=float, default=.95)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--collector-pool', type=Path)
    parser.add_argument('--collector-slots', type=int, default=4)
    parser.add_argument('--experiments', type=int, default=3)
    parser.add_argument('--ablations', type=int, default=2)
    args = parser.parse_args()
    if bool(args.episode_ids) != bool(args.episode_ids_sha256):
        parser.error('episode-ids requires episode-ids-sha256')
    if not 1 <= args.collector_slots <= 16 or not 0 <= args.low < args.high <= 1 or not 1 <= args.workers <= 16 or args.experiments < 1 or not 0 <= args.ablations <= 12:
        parser.error('invalid band or collection allocation')
    stop = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        if args.episode_ids and args.screen_out.exists():
            if digest(args.episode_ids) != args.episode_ids_sha256:
                raise ValueError('predeclared episode batch changed')
            ids = json.loads(args.episode_ids.read_text())
            try:
                completed = {row['episode'] for row in rows(args.screen_out)}
            except (json.JSONDecodeError, UnicodeDecodeError):  # a live append is not a complete snapshot yet
                completed = set()
            if set(ids) <= completed:
                break
        state = subprocess.check_output(['systemctl', '--user', 'show', args.screen_service,
            '-p', 'ActiveState', '-p', 'LoadState', '-p', 'ExecMainStatus', '-p', 'Result'], text=True)
        fields = dict(line.split('=', 1) for line in state.splitlines() if '=' in line)
        if fields.get('ActiveState') in {'active', 'activating', 'deactivating'}:
            stop.wait(30)
            continue
        # Successful transient services may be unloaded before this poll. Complete pinned
        # screen/kept artifacts still undergo the full independent gate below.
        if fields.get('LoadState') == 'not-found' and args.kept.is_file():
            break
        if fields.get('ActiveState') != 'inactive' or fields.get('Result') != 'success' or fields.get('ExecMainStatus') != '0':
            raise ValueError('screen service did not finish successfully: ' + state)
        break
    if stop.is_set():
        return
    receipt = validate_handoff(args)
    if stop.is_set():
        return
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'screen-handoff.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('handoff resume identity changed')
    write_json(path, receipt)
    print(json.dumps({'screened': receipt['screened'], 'selected': receipt['selected']}), flush=True)
    if not receipt['selected']:
        return
    args.episodes = args.kept
    args.database_root = args.arena_root = None
    args.max_attempts = 1
    args.backoff_seconds = 30
    print(json.dumps(run_queue(args)), flush=True)


if __name__ == '__main__':
    main()
