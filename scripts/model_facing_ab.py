#!/usr/bin/env python3
"""Paired live A/B of model-facing runtime variants on a served policy (AGENTS.md "Measure model-facing changes live").

Reuses the RL task-rollout pipeline: queue entries from `rl.task_rollouts.build_queue`, the production collector
(`scripts/run_bonsai_queue.py`, ts-host dist/teacher/cli.js) against an already-served OpenAI-compatible policy, and the
production classifier (`rl.task_rollouts.score`). Arms differ only in the `NATLANG_MODEL_FACING` environment variable of
the collector (`ts-host/src/runtime/model-facing.ts`); every arm gets the same tasks, seeds and sample order (sample-major,
so a partly finished run stays balanced), and the arms run concurrently so server load is shared.

  prepare  --packet P --rows 3,14,15 --samples 8 --arms baseline=none,combined=combined --out DIR
  run      --out DIR --server http://127.0.0.1:8200 [--runtime RT]
  report   --out DIR

Variants of one arm are joined with `+` (`--arms base=effect-receipt+read-file-binding,other=effect-receipt+read-file-binding+<new-variant>`);
the first arm is the reference the others are paired against. Training-split packet rows only (prepare refuses any other split).

A policy actor of its own on Pop's idle GPU (`serve`, `guard`, `stop`): the Pop RL actor's machinery (the policy's prepared
view, the evaluation image, `pop_dev16_driver.py` in rl-actor mode) under a container, unit, work directory and port of
this measurement, without the production hooks' state or marker files. `guard` stops the container and the collectors as
soon as an evaluation is pending or a Pop RL actor of the controller exists: the production use of Pop always wins.

  serve    --checkpoint-sha256 SHA [--port 18541]      (writes DIR-independent state under the Pop evaluation root)
  guard    --out DIR [--interval 30]                   (loop; run it beside `run`)
  stop
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.rl import task_rollouts  # noqa: E402
from natlang_neuralese.eval.task_packet import read_jsonl  # noqa: E402


def prepare(args) -> None:
    out = args.out.resolve()
    rows = [read_jsonl(args.packet)[int(index)] for index in args.rows.split(',')]
    if any(row.get('split') != 'train' for row in rows):
        raise SystemExit('only training-split rows may be measured')
    arms = dict(item.split('=', 1) for item in args.arms.split(','))
    for arm, variant in arms.items():
        directory = out / arm
        directory.mkdir(parents=True, exist_ok=True)
        packet = directory / 'packet.jsonl'
        packet.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        entries = task_rollouts.build_queue(packet, len(rows), args.samples, directory, temperature=args.temperature,
                                            case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                                            max_turns=args.max_turns, seed_base=args.seed_base)
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=32768)
        entries.sort(key=lambda entry: (entry['sample'], entry['index']))
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries))
        (directory / 'preparation.json').write_text(json.dumps({
            'schema': 'natlang.task-rollout-preparation/1', 'root_mode': 'student', 'model_facing': variant,
            'source_packet': str(args.packet), 'source_rows': args.rows,
            'packet_sha256': hashlib.sha256(packet.read_bytes()).hexdigest()}, indent=2) + '\n')
    (out / 'arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    print(json.dumps({'arms': arms, 'tasks': [row['id'] for row in rows], 'entries_per_arm': len(rows) * args.samples}))


def run(args) -> None:
    out = args.out.resolve()
    arms = json.loads((out / 'arms.json').read_text())
    config = out / 'sampling-config.json'
    config.write_text(json.dumps({'guidance': False}) + '\n')
    children = []
    for arm, variant in arms.items():
        env = dict(os.environ, NATLANG_MODEL_FACING=variant.replace('+', ','))
        log = (out / arm / 'queue.log').open('a')
        command = [sys.executable, str(ROOT / 'scripts' / 'run_bonsai_queue.py'), str(out / arm / 'queue.jsonl'),
                   str(out / arm / 'journal.jsonl'), '--runtime', str(args.runtime), '--server', args.server,
                   '--chat-completions-url', args.server.rstrip('/') + '/v1/chat/completions', '--model-id', args.model_id,
                   '--model-concurrency', '1', '--chat-request-config', str(config), '--neuralese-server']
        children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    (out / 'pids.json').write_text(json.dumps([child.pid for child in children]) + '\n')
    print(json.dumps({'pids': [child.pid for child in children]}), flush=True)
    for child in children:
        child.wait()


STUB_WORDS = re.compile(r'Decide whether item meets policy|Refunds are allowed within 30 days')
INLINE_NL = re.compile(r'\bnl(?:\.with)?(?:<[^`(]*>)?(?:\([^`]*\))?`')
NL_TEMPLATE = re.compile(r'\bnl(?:\.with)?(?:<[^`(]*>)?(?:\([^`]*\))?`(?:[^`\\]|\\.)*`(\s*\()?')
ZERO_ARG_CHILD = re.compile(r'\bnl(?:\.with)?(?:<[^`(]*>)?(?:\([^`]*\))?`[^`]*`\(\s*\)')


def behaviours(directory: Path) -> dict:
    """What each change targets, counted over the arm's actual actions (each action and tool result counted once)."""
    counts = dict(episodes=0, turns=0, evals=0, repeated_eval_code=0, zero_arg_children=0,
                  prompt_example_copies=0, receipts=0, footers=0, read_file_calls=0,
                  inline_nl_calls=0, nl_not_invoked=0, episodes_with_inline_nl=0, nl_child_quiesced=0)
    for results in sorted(directory.glob('row-*/sample-*/results.jsonl')):
        for row in read_jsonl(results):
            counts['episodes'] += 1
            codes = {}
            nl_before = counts['inline_nl_calls']
            for turn in row.get('trajectory') or []:
                if turn.get('phase') != 'action':
                    continue
                counts['turns'] += 1
                context = turn.get('context') or []
                last = context[-1] if context else {}
                if last.get('role') == 'tool':  # the result of the previous action, seen once per turn
                    result = str(last.get('content'))
                    counts['nl_child_quiesced'] += 'was itself created with these same instructions' in result
                    counts['receipts'] += 'Service calls completed by this eval' in result
                    counts['footers'] += '(in eval: const text = await folder.file(' in result
                for name, args in (turn.get('model_response') or {}).get('calls') or []:
                    counts['read_file_calls'] += name == 'read_file'
                    if name != 'eval':
                        continue
                    code = str((args or {}).get('code', ''))
                    counts['evals'] += 1
                    counts['prompt_example_copies'] += len(STUB_WORDS.findall(code))
                    counts['zero_arg_children'] += len(ZERO_ARG_CHILD.findall(code))
                    counts['inline_nl_calls'] += len(INLINE_NL.findall(code))
                    counts['nl_not_invoked'] += sum(not match.group(1) for match in NL_TEMPLATE.finditer(code))
                    key = (turn.get('invocation_id'), code)
                    counts['repeated_eval_code'] += key in codes
                    codes[key] = True
            counts['episodes_with_inline_nl'] += counts['inline_nl_calls'] > nl_before
    return counts


def report(args) -> None:
    out = args.out.resolve()
    arms = json.loads((out / 'arms.json').read_text())
    summary = {}
    for arm in arms:
        score = task_rollouts.score(out / arm)
        rewards = [json.loads(line) for line in (out / arm / 'rewards.jsonl').read_text().splitlines()]
        finished = [row for row in rewards if row['finish_status'] == 'complete']
        by_task = {}
        for row in finished:
            by_task.setdefault(row['task_id'][:60], []).append(row['outcome'] == 'task_success')
        summary[arm] = {'variant': arms[arm], 'finished': len(finished), 'successes': sum(row['outcome'] == 'task_success'
                        for row in finished), 'outcomes': score['outcome_counts'],
                        'by_task': {task: f'{sum(v)}/{len(v)}' for task, v in by_task.items()},
                        'behaviours': behaviours(out / arm)}
    summary['paired'] = paired(out, list(arms))
    print(json.dumps(summary, indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


def paired(out: Path, arms: list[str]) -> dict:
    """Each later arm against the first, over the matched (task, sample) episodes finished in both: successes,
    discordant pairs, McNemar exact p, turns and tokens."""
    rows = {}
    for arm in arms:
        rewards = {(r['task_id'], r['sample']): r for r in map(json.loads, (out / arm / 'rewards.jsonl').read_text().splitlines())}
        finishes = {}
        for line in (out / arm / 'journal.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'finish' and event.get('status') == 'complete':  # an incomplete run is infrastructure, not a result
                finishes[event['key']] = event
        keys = {e['key']: (e['task_id'], e['sample']) for e in map(json.loads, (out / arm / 'queue.jsonl').read_text().splitlines())}
        rows[arm] = {keys[k]: (rewards[keys[k]]['outcome'] == 'task_success', f.get('saved_turns'), f.get('completion_tokens'),
                              f.get('elapsed_seconds')) for k, f in finishes.items()}
    first, result = arms[0], {}
    for second in arms[1:]:
        common = sorted(set(rows[first]) & set(rows[second]))
        only_first = sum(rows[first][k][0] and not rows[second][k][0] for k in common)
        only_second = sum(rows[second][k][0] and not rows[first][k][0] for k in common)
        discordant = only_first + only_second
        p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(only_first, only_second) + 1)) / 2 ** discordant) if discordant else 1.0
        mean = lambda arm, i: round(sum(rows[arm][k][i] or 0 for k in common) / max(len(common), 1), 1)
        result[f'{first}_vs_{second}'] = {
            'pairs': len(common), f'{first}_successes': sum(rows[first][k][0] for k in common),
            f'{second}_successes': sum(rows[second][k][0] for k in common),
            f'only_{first}': only_first, f'only_{second}': only_second, 'mcnemar_exact_p': round(p, 3),
            f'{first}_mean_turns': mean(first, 1), f'{second}_mean_turns': mean(second, 1),
            f'{first}_mean_tokens': mean(first, 2), f'{second}_mean_tokens': mean(second, 2)}
    return result


# ---- A policy actor of its own on Pop ----------------------------------------------------------------------------
AB_PORT = 18541


def _pop():
    sys.path.insert(0, str(ROOT))
    from scripts.neuralese_data import pop_behavior_eval as pbe
    return pbe


def _ab_state(pbe) -> Path:
    return pbe.ROOT / 'model-facing-ab-actor.json'   # not a production hook file (rl-actor.json, rl-actor-waiter.json, rl-actor-active.json)


def pop_busy(pbe, own: str | None = None) -> str | None:
    """Why Pop belongs to production right now (a dev16 evaluation pending, a Pop RL actor started or deferred by the
    controller, any other container), or None."""
    state = pbe.read_json(pbe.ROOT / 'state.json') or {'candidates': {}, 'controllers': {}, 'parent': {'sha256': ''}}
    if pbe.eval_pending(state):
        return 'a dev16 evaluation is pending'
    for marker in ('rl-actor-active.json', 'rl-actor-waiter.json', 'rl-actor.json'):
        if (pbe.ROOT / marker).exists():
            return f'the controller wrote {marker}'
    names = pbe.ssh("docker ps --format '{{.Names}}'", check=True).split()
    foreign = [name for name in names if name != own]
    return f'Pop containers {foreign} are running' if foreign else None


def serve(args) -> None:
    import shlex
    import time
    pbe = _pop()
    from scripts.neuralese_data import pop_rl_actor as pra
    sha = args.checkpoint_sha256
    if reason := pop_busy(pbe):
        raise SystemExit('Pop is not free: ' + reason)
    prep = pbe.read_json(pbe.ROOT / 'preparations' / f'rl-{sha}.json')
    if not prep or not (Path(prep['transport']['path']).parent / prep['view']['path']).is_file():
        raise SystemExit('no prepared Pop view of this policy; start the controller hook once or run pop_eval_prepare')
    transport = json.loads(Path(prep['transport']['path']).read_text())
    stage = f'{pbe.POP_TRANSPORT_ROOT}/{sha}'
    have = pbe.ssh(shlex.join(['ls', stage]), check=False).split()
    if prep['view']['path'] not in have:   # the view is not on Pop yet: room first, then the exact delta
        pbe.make_pop_room(pbe.read_json(pbe.ROOT / 'state.json'), prep['view']['bytes'], {sha})
        pbe.send(prep)
    commit = pbe.ssh(shlex.join(['git', '-C', pbe.POP_REPO, 'rev-parse', 'HEAD'])).strip()
    dirty = int(pbe.ssh(shlex.join(['git', '-C', pbe.POP_REPO, 'status', '--porcelain', '--untracked-files=no']) + ' | wc -l')) > 0
    host = pra.pop_host_address()
    endpoint = f'http://{host}:{args.port}'
    name, work = f'popab-{sha[:12]}', f'{pbe.POP_EVAL_ROOT}/popab-{sha[:12]}'
    job = {'mode': 'rl-actor', 'arm': 'rl', 'name': name, 'port': args.port, 'host': host, 'runtime': pbe.POP_REPO,
           'code': {'git_commit': commit, 'dirty': dirty}, 'base': pbe.POP_BASE, 'image': pbe.IMAGE, 'step': prep['step'],
           'source_sha256': sha, 'stage': stage, 'view': prep['view']['path'], 'view_sha256': prep['view']['sha256'],
           'work': work, 'min_free_bytes': pbe.POP_MIN_FREE, 'actor_export_format': pra.EXPORT_FORMAT,
           'memory': f'{pra.RL_MEMORY_GB}g',
           'transport': {'name': Path(prep['transport']['path']).name, 'sha256': prep['transport']['sha256'],
                         'view_receipt_name': prep['view_receipt']['sha256'] + '.json',
                         'view_receipt_sha256': prep['view_receipt']['sha256'], 'scope': transport['scope']}}
    pbe.ssh(shlex.join(['mkdir', '-p', work]) + ' && ' + shlex.join(['rm', '-f', f'{work}/driver.py']))
    subprocess.run(['scp', '-q', str(ROOT / 'scripts/neuralese_data/pop_dev16_driver.py'), f'{pbe.POP}:{work}/driver.py'],
                   check=True, timeout=120)
    pbe._pop_upload_json(job, f'{work}/job.json')
    unit = f'{name}-{int(time.time())}'
    state = {'endpoint': endpoint, 'unit': unit, 'container': name, 'policy_id': sha, 'work': work, 'port': args.port}
    _ab_state(pbe).write_text(json.dumps(state, indent=1) + '\n')
    pbe.ssh(shlex.join(['/usr/bin/python3', pbe.POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                        '--budget-gb', str(pra.RL_MEMORY_GB), '--reserve-gb', str(pbe.POP_RESERVE_GB), '--class', 'experiment',
                        '--no-resumable', '--container', name, '--workdir', work, '--', '/usr/bin/python3',
                        f'{work}/driver.py', f'{work}/job.json']))
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if pbe.pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed'):
            tail = pbe.ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '20', '--no-pager']), check=False)
            raise SystemExit('Pop actor unit exited before readiness: ' + tail[-1200:])
        if pra.attested(endpoint, sha):
            print(json.dumps({**state, 'ready': True}), flush=True)
            return
        time.sleep(5)
    raise SystemExit(f'Pop actor did not attest {sha[:12]} at {endpoint} within {args.ready_seconds} s')


def stop(_args=None) -> None:
    """Stop this measurement's actor and remove its work directory (root-owned files, so from a throwaway container)."""
    import shlex
    pbe = _pop()
    path = _ab_state(pbe)
    state = pbe.read_json(path)
    if not state:
        return
    pbe.ssh(shlex.join(['systemctl', '--user', 'stop', state['unit'] + '.service']), check=False, timeout=120)
    pbe.ssh(shlex.join(['docker', 'rm', '-f', state['container']]), check=False, timeout=60)
    pbe.remove_pop_work([state['work']])
    path.unlink(missing_ok=True)
    print(json.dumps({'stopped': state['container'], 'removed': state['work']}), flush=True)


def guard(args) -> None:
    """Hand Pop back: stop the actor and the collectors as soon as production needs it."""
    import signal
    import time
    pbe = _pop()
    pids = json.loads((args.out / 'pids.json').read_text())
    state = pbe.read_json(_ab_state(pbe))
    while True:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        reason = pop_busy(pbe, own=state['container'])
        if reason or not alive:
            if reason:
                for pid in alive:
                    os.kill(pid, signal.SIGTERM)
                (args.out / 'guard-trip.json').write_text(json.dumps({'reason': reason, 'at': time.time()}) + '\n')
            stop()
            print(json.dumps({'guard': reason or 'collectors finished'}), flush=True)
            return
        time.sleep(args.interval)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--rows', required=True, help='comma-separated packet row indices')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--arms', default='baseline=none,combined=combined')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--case-seconds', type=int, default=1800)
    p.add_argument('--max-model-requests', type=int, default=64)
    p.add_argument('--max-turns', type=int, default=128)
    p.add_argument('--seed-base', type=int, default=4100000)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--server', required=True)
    r.add_argument('--runtime', type=Path, default=ROOT / 'ts-host')
    r.add_argument('--model-id', default='natlang-rl')
    q = sub.add_parser('report')
    q.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('serve')
    v.add_argument('--checkpoint-sha256', required=True)
    v.add_argument('--port', type=int, default=AB_PORT)
    v.add_argument('--ready-seconds', type=int, default=900)
    g = sub.add_parser('guard')
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--interval', type=int, default=30)
    sub.add_parser('stop')
    args = parser.parse_args(argv)
    {'prepare': prepare, 'run': run, 'report': report, 'serve': serve, 'guard': guard, 'stop': stop}[args.command](args)


if __name__ == '__main__':
    main()
