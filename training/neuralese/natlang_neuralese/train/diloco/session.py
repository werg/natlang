"""Prepare an elastic DiLoCo session from an earned joint-SFT window boundary (DGX side).

    session.py prepare --boundary-config W0/config.json --session-dir DIR --run-id ID [--workers 8] [--cohost-worker 1]
                       [--window 512] [--local-steps 64] [--relay-port 18080] [--coordinator-port 8080]
        (run under the lineage's frozen runtime: PYTHONPATH=RUNTIME/training/neuralese:RUNTIME, cwd RUNTIME)
    session.py relocate --session-dir DIR --worker K --window-dir WDIR --parent P --id ID --prior-out OUT
        (run with the main checkout: rl_cloud.relocate_control, exactly what a prepared-control seed applies)

``prepare`` builds the one session control every worker runs:

1. the lineage's own next window from the earned boundary (``sft_window.py next --window W``, the frozen runtime's
   ``continuation_config``: same recipe, objective, focus, optimizer/RNG/exposure and scored-graph anchor),
2. the recipe stage's ``diloco`` block (identical for every worker: run, slot count, local steps, the local relay URL;
   the worker id is the env ``NATLANG_DILOCO_WORKER_ID`` set by join.py), written as DIR/recipe-runtime.json outside
   the window directory so it ships at one path,
3. ``seed_command`` wrapped by join.py (``--worker K`` is the only per-worker difference),
4. one control per worker slot (DIR/workers/worker-K/config.json, never started; a cloud lineage adopts it with
   ``prepared_control``), and DIR/session.json plus DIR/bin/{join,relay}.py.

``relocate`` writes worker K's control relocated to an existing lineage's next window (the running learner's
in-place join): ``branch_origin.prepared_next_window`` binds the previous window directory and the earned parent's
sha256 so that ``sft_window.py next`` adopts it at that boundary and refuses it anywhere else.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSION_SCHEMA = 'natlang.diloco-elastic-session/1'
DEFAULT_SCHEDULE = [{'from_round': 0, 'lr': 1.0, 'momentum': 0.0}, {'from_round': 10, 'lr': 0.7, 'momentum': 0.9}]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f'{path} exists: session artifacts are immutable (write a new session)')
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    return path


def unwrap_join(command):
    """The trainer command under any earlier session's join.py wrapper. A session prepared from a window of a previous
    session (s3 from d1's s2 boundary, 2026-11-04) otherwise nests both joins, and the inner one rejects the new
    session's --diloco block."""
    command = list(command)
    while len(command) > 2 and Path(command[1]).name == 'join.py' and '--' in command:
        command = command[command.index('--') + 1:]
    return command


def prepare(args):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.recipe import load_recipe, native_recipe_binding
    directory = Path(args.session_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f'{directory} is not empty')
    prep = directory / 'prep'
    prep.mkdir(parents=True)
    runtime = os.getcwd()  # the lineage's frozen runtime, as lfm3_rl_lineage.sh runs sft_window.py
    window_tool = Path(args.window_tool).resolve()
    subprocess.run([sys.executable, str(window_tool), 'next', '--previous-config', args.boundary_config,
                    '--out', str(prep / 'out'), '--config-out', str(prep / 'config.json'), '--window', str(args.window),
                    '--id', f'{args.run_id}-session', *(['--runtime', args.runtime] if args.runtime else []),
                    *(['--trainer-parameter-request', args.trainer_parameter_request]
                      if args.trainer_parameter_request else [])],
                   cwd=runtime, check=True)
    base = json.loads((prep / 'config.json').read_text())
    stage_id = base['production_stage']
    old_recipe = Path(base['production_recipe'])
    # --recipe: a successor recipe (e.g. one that adds interleaved RL stages) replaces the window's own copy; its
    # lineage must still bind the control's recipe_lineage_sha256 (native_recipe_binding below refuses otherwise).
    recipe = json.loads(Path(args.recipe or old_recipe).read_text())
    stage = next(s for s in recipe['stages'] if s['id'] == stage_id)
    block = {'run_id': args.run_id, 'workers': args.workers, 'coordinator_url': f'http://127.0.0.1:{args.relay_port}',
             'local_steps': args.local_steps, 'start_grace_seconds': args.start_grace_seconds,
             'timeout_seconds': args.timeout_seconds}
    if stage['parameters'].get('refresh_after_updates') != args.window:
        raise SystemExit('the prepared window did not adopt the session window length')
    stage['parameters']['diloco'] = block
    # DiLoCo learners keep no local weight EMA (native_state_sft refuses one); the coordinator keeps the global EMA.
    stage['parameters']['weight_ema'] = 'off'
    # Inputs the prepared window wrote under its own directory (control-inputs snapshots, which bind each other's
    # paths) stay at that session path, which ships to every worker; only the window's own outputs relocate.
    text = json.dumps(base)
    moved = sorted(str(p.relative_to(prep / 'out')) for p in (prep / 'out').rglob('*') if p.is_file() and p != old_recipe)
    recipe_path = write(directory / 'recipe-runtime.json', recipe)
    load_recipe(recipe_path)
    binding = native_recipe_binding(recipe_path, base['recipe_lineage_sha256'])
    text = text.replace(json.dumps(str(old_recipe))[1:-1], str(recipe_path))
    control = json.loads(text)
    control['production_recipe'] = str(recipe_path)
    for key in ('trainer_parameters', 'seed_trainer_parameters'):
        control[key]['diloco'] = block
        control[key]['weight_ema'] = 'off'
        control[key]['recipe_binding'] = binding
    old_recipe.unlink()
    if any(not x.startswith('control-inputs/') for x in moved):
        raise SystemExit(f'the prepared window wrote unexpected inputs in its directory {moved}')
    bin_dir = directory / 'bin'
    bin_dir.mkdir()
    for name in ('join.py', 'relay.py'):
        shutil.copy2(HERE / name, bin_dir / name)
    session = {'schema': SESSION_SCHEMA, 'run_id': args.run_id, 'workers': args.workers,
               'cohost_worker': args.cohost_worker, 'relay_port': args.relay_port, 'relay': str(bin_dir / 'relay.py'),
               'state_root': args.state_root,
               'coordinator': {'root': f'{args.state_root}/{args.run_id}/root', 'port': args.coordinator_port,
                               'code': args.coordinator_code, 'python': args.python,
                               'init': ['--quorum', '1', '--max-staleness', str(args.max_staleness),
                                        '--grace-seconds', str(args.grace_seconds),
                                        '--outer-schedule', json.dumps(DEFAULT_SCHEDULE)]},
               'window_inputs': moved,
               'boundary': {'config': str(Path(args.boundary_config).resolve()),
                            'parent': base['branch_origin']['parent'], 'window_updates': args.window,
                            'local_steps': args.local_steps}}
    session_path = write(directory / 'session.json', session)
    seed = unwrap_join(control['seed_command'])
    for worker in range(args.workers):
        item = copy.deepcopy(control)
        # Every file join.py runs is named in the control itself, so the lineage closure ships it.
        item['seed_command'] = [seed[0], str(bin_dir / 'join.py'), '--session', str(session_path),
                                '--relay', str(bin_dir / 'relay.py'), '--worker', str(worker), '--', *seed]
        item['out'] = str(directory / 'workers' / f'worker-{worker}' / 'out')
        text = json.dumps(item).replace(json.dumps(str(prep / 'out') + '/resident-data')[1:-1],
                                        item['out'] + '/resident-data')
        item = json.loads(text)
        if json.dumps(item).count(str(prep / 'out') + '/') != json.dumps(item).count(str(prep / 'out') + '/control-inputs/'):
            raise SystemExit('a worker control still names the preparation window directory')
        item['id'] = f'{args.run_id}-worker-{worker}'
        item['branch_origin'] = {**item['branch_origin'], 'diloco_session': {
            'session': str(session_path), 'worker': worker, 'run_id': args.run_id,
            'session_sha256': sha256_file_hex_cached(session_path)}}
        write(directory / 'workers' / f'worker-{worker}' / 'config.json', item)
    print(json.dumps({'session': str(session_path), 'workers': args.workers, 'recipe': binding,
                      'parent': base['branch_origin']['parent'], 'controls': str(directory / 'workers')}, indent=1))


def relocate(args):
    sys.path.insert(0, str(HERE.parents[4] / 'scripts' / 'coreweave'))
    import rl_cloud
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    directory = Path(args.session_dir).resolve()
    source = json.loads((directory / 'workers' / f'worker-{args.worker}' / 'config.json').read_text())
    config = rl_cloud.relocate_control(source, Path(args.window_dir), Path(args.parent), args.id)
    parent_sha = sha256_file_hex_cached(Path(args.parent))
    if parent_sha != source['branch_origin']['parent']['sha256']:
        raise SystemExit('the lineage parent differs from the session seed')
    config['branch_origin']['prepared_next_window'] = {'prior_out': os.path.abspath(args.prior_out),
                                                       'parent_sha256': parent_sha}
    out = Path(args.output or (directory / 'relocated' / f'worker-{args.worker}' / 'config.json'))
    write(out, config)
    print(json.dumps({'config': str(out), 'out': config['out'], 'checkpoint': config['checkpoint']}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--boundary-config', required=True)
    p.add_argument('--session-dir', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cohost-worker', type=int, default=1)
    p.add_argument('--window', type=int, default=512)
    p.add_argument('--local-steps', type=int, default=64)
    p.add_argument('--start-grace-seconds', type=float, default=60.0)
    p.add_argument('--timeout-seconds', type=float, default=300.0)
    p.add_argument('--relay-port', type=int, default=18080)
    p.add_argument('--coordinator-port', type=int, default=8080)
    p.add_argument('--max-staleness', type=int, default=2)
    p.add_argument('--grace-seconds', type=float, default=300.0)
    p.add_argument('--state-root', default='/work/diloco')
    p.add_argument('--coordinator-code', help='the bundle code the cohost serves the coordinator from '
                   '(default: the machine profile repo root, the same path in every sandbox)')
    p.add_argument('--python', help='default: this (the learner\'s) python')
    p.add_argument('--recipe', help='successor recipe to bind instead of the prepared window\'s own copy')
    p.add_argument('--runtime', help='frozen runtime for the session window (sft_window next --runtime)')
    p.add_argument('--trainer-parameter-request', help='memory/composition trainer parameters for the session window '
                   '(sft_window next --trainer-parameter-request)')
    p.add_argument('--window-tool', default=str(HERE.parents[4] / 'scripts' / 'neuralese_data' / 'sft_window.py'))
    r = commands.add_parser('relocate')
    r.add_argument('--session-dir', required=True)
    r.add_argument('--worker', type=int, required=True)
    r.add_argument('--window-dir', required=True)
    r.add_argument('--parent', required=True)
    r.add_argument('--prior-out', required=True)
    r.add_argument('--id', required=True)
    r.add_argument('--output')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from natlang_neuralese.common.paths import resolve_str
        args.coordinator_code = args.coordinator_code or resolve_str('repo')
        args.python = args.python or sys.executable  # frozen runtimes carry no venv; the learner's own python
    {'prepare': prepare, 'relocate': relocate}[args.command](args)


if __name__ == '__main__':
    main()
