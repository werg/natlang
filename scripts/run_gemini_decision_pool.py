#!/usr/bin/env python3
"""Rotate held typed-decision generation across verified free Google text models.

One owner per project state file; aliases share quota groups. Rate-limit attempts
are journaled without marking the case completed. Existing strict answer checks
and gold-free prompts come from label_decision_cases, not a second adapter.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from label_decision_cases import _http_teacher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common import jsonio  # noqa: E402

write_json = jsonio.write_json_durable

ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--models', default='training/providers/gemini-free-text-pool.json')
    p.add_argument('--state', required=True, help='shared project quota state, also locked against duplicate owners')
    p.add_argument('--api-key-env', default='GEMINI_API_KEY')
    p.add_argument('--exclude-model', action='append', default=[], help='reserve this model quota group for an existing worker')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--choice-contract', choices=['probabilities', 'label-confidence'], default='probabilities',
                   help='choice response shape; score/noul cases retain the probability contract')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--workers', type=int, default=4, help='parallel quota groups; at most one in-flight call per group')
    p.add_argument('--wait', action='store_true', help='wait for cooldowns instead of exiting with unfinished cases')
    p.add_argument('--wait-for-owner', action='store_true', help='queue this source behind the current project pool owner')
    args = p.parse_args()
    if args.limit < 0 or args.timeout <= 0 or args.workers < 1:
        p.error('limit must be nonnegative and timeout positive')
    key = os.environ.get(args.api_key_env)
    if not key:
        p.error('API key environment variable is unset or empty')
    config = json.loads(Path(args.models).read_text())
    models = config['models']
    if not models or len({m['id'] for m in models}) != len(models):
        p.error('pool requires distinct model IDs')
    if any(m.get('free_text_verified') is not True or float(m['interval_seconds']) <= 0 for m in models):
        p.error('pool only accepts explicitly verified free text models with positive pacing')
    excluded = {m['quota_group'] for m in models if m['id'] in args.exclude_model}
    models = [m for m in models if m['quota_group'] not in excluded]
    if not models:
        p.error('all model quota groups excluded')
    state_path, out_path = Path(args.state), Path(args.out)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(state_path) + '.lock', 'a')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not args.wait_for_owner:
                p.error('another pool owns this project state; use its queue, not another state file')
            print(json.dumps({'event': 'waiting_for_project_owner'}), flush=True)
            time.sleep(30)
    state = json.loads(state_path.read_text()) if state_path.exists() else {'groups': {}, 'cursor': 0}
    groups = state.setdefault('groups', {})
    identity = {'schema': 'natlang.gemini-decision-pool/1', 'training_admission': False,
                'cases_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
                'choice_contract': args.choice_contract,
                'models': models, 'endpoint': ENDPOINT, 'workers': args.workers,
                'adapter_sha256': hashlib.sha256(Path(__file__).with_name('label_decision_cases.py').read_bytes()).hexdigest(),
                'scheduler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'persistence_sha256': hashlib.sha256(Path(jsonio.__file__).read_bytes()).hexdigest()}
    manifest = Path(str(out_path) + '.manifest.json')
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        p.error('output provenance changed; select a new output')
    if not manifest.exists():
        write_json(manifest, identity)
    done = {json.loads(line)['id'] for line in out_path.read_text().splitlines() if line.strip()} if out_path.exists() else set()
    cases = deque(json.loads(line) for line in Path(args.cases).read_text().splitlines()
                  if line.strip() and json.loads(line)['id'] not in done)
    if args.limit:
        cases = deque(list(cases)[:args.limit])
    count, pending = 0, {}
    def ask(case, model):
        return _http_teacher(case, endpoint=ENDPOINT, model=model['id'], api_key=key,
            timeout=args.timeout, retries=0, initial_backoff=30, max_backoff=300,
            reasoning_effort=model['reasoning_effort'], max_output_tokens=512,
            response_format='json_schema', choice_contract=args.choice_contract)
    with ThreadPoolExecutor(max_workers=args.workers) as executor, out_path.open('a') as out, open(str(out_path) + '.attempts.jsonl', 'a') as journal:
        while cases or pending:
            now = time.time()
            busy = {model['quota_group'] for case, model in pending.values()}
            order = [(state.get('cursor', 0) + i) % len(models) for i in range(len(models))]
            for index in order:
                if not cases or len(pending) >= args.workers:
                    break
                model = models[index]
                group = groups.setdefault(model['quota_group'], {})
                if model['quota_group'] in busy or group.get('disabled') or group.get('not_before', 0) > now:
                    continue
                case = cases.popleft()
                state['cursor'] = (index + 1) % len(models)
                group['not_before'] = now + model['interval_seconds']
                write_json(state_path, state)
                pending[executor.submit(ask, case, model)] = (case, model)
                busy.add(model['quota_group'])
            if not pending:
                future = [groups.get(m['quota_group'], {}).get('not_before', 0) for m in models
                          if not groups.get(m['quota_group'], {}).get('disabled')]
                if not args.wait or not future:
                    print(json.dumps({'event': 'pool_waiting', 'unfinished_cases': len(cases),
                                      'not_before': min(future) if future else None}), flush=True)
                    return
                time.sleep(min(60, max(0.1, min(future) - time.time())))
                continue
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for task in completed:
                case, model = pending.pop(task)
                try:
                    answer, provider = task.result()
                except Exception as exc:
                    # A programming failure is diagnosable, never a fabricated label.
                    journal.write(json.dumps({'id': case['id'], 'model': model['id'],
                                              'exception': type(exc).__name__}) + '\n')
                    journal.flush()
                    raise
                journal.write(json.dumps({'id': case['id'], 'time': time.time(), 'answer': answer, 'provider': provider}) + '\n')
                journal.flush()
                group = groups[model['quota_group']]
                status = provider['http_status']
                if status == 429 or status == 0 or status >= 500:
                    group['failure_streak'] = group.get('failure_streak', 0) + 1
                    delays = [h.get('retry_after_seconds', 0) for h in provider['retry_history']]
                    quota_ids = [q for h in provider['retry_history'] for q in h.get('quota_ids', [])]
                    delay = max(max(delays, default=0), min(300, 15 * 2 ** min(group['failure_streak'] - 1, 5)))
                    if any('PerDay' in q for q in quota_ids) and not any(delays):
                        group['disabled'] = 'daily_quota_without_retry_info_requires_review'
                    group.update(not_before=time.time() + delay, quota_ids=quota_ids)
                    cases.appendleft(case)
                    print(json.dumps({'event': 'model_cooldown', 'model': model['id'], 'seconds': delay}), flush=True)
                elif status in (400, 401, 403, 404):
                    group['disabled'] = f'http_{status}_requires_review'
                    cases.appendleft(case)
                    print(json.dumps({'event': 'model_disabled', 'model': model['id'], 'status': status}), flush=True)
                else:
                    group['failure_streak'] = 0
                    provider['cases_sha256'] = identity['cases_sha256']
                    contract = ('natlang.choice-label-confidence/1'
                                if case.get('kind') == 'choice' and args.choice_contract == 'label-confidence'
                                else 'natlang.typed-decision-probabilities/1')
                    out.write(json.dumps({'id': case['id'], 'family': case['family'],
                        'teacher': 'google/' + model['id'], 'decision_contract': contract,
                        'answer': answer, 'provider': provider}) + '\n')
                    out.flush()
                    count += 1
                    print(json.dumps({'labelled': count, 'model': model['id'], 'error': answer.get('error')}), flush=True)
                write_json(state_path, state)


if __name__ == '__main__':
    main()
