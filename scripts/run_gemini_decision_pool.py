#!/usr/bin/env python3
"""Rotate held typed-decision generation across verified free Google text models.

One owner per project state file; aliases share quota groups. Rate-limit attempts
are journaled without marking the case completed. Existing strict answer checks
and gold-free prompts come from label_decision_cases, not a second adapter.
"""
import argparse
from collections import Counter, deque
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
DAILY_QUOTA_NO_RETRY_REASON = 'daily_quota_without_retry_info_requires_review'
DAILY_QUOTA_REPROBE_BASE_SECONDS = 60 * 60
DAILY_QUOTA_REPROBE_MAX_SECONDS = 24 * 60 * 60
DAILY_QUOTA_REPROBE_HISTORY_LIMIT = 32


def _append_daily_reprobe_event(record, event):
    history = record.setdefault('history', [])
    history.append(event)
    if len(history) > DAILY_QUOTA_REPROBE_HISTORY_LIMIT:
        del history[:-DAILY_QUOTA_REPROBE_HISTORY_LIMIT]


def _finish_daily_reprobe(group, record, event):
    _append_daily_reprobe_event(record, event)
    history = group.setdefault('daily_quota_reprobe_history', [])
    history.append(record)
    if len(history) > DAILY_QUOTA_REPROBE_HISTORY_LIMIT:
        del history[:-DAILY_QUOTA_REPROBE_HISTORY_LIMIT]
    group.pop('daily_quota_reprobe', None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--models', default='training/providers/gemini-free-text-pool.json')
    p.add_argument('--state', required=True, help='shared project quota state, also locked against duplicate owners')
    p.add_argument('--api-key-env', default='GEMINI_API_KEY')
    p.add_argument('--exclude-model', action='append', default=[], help='reserve this model quota group for an existing worker')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-output-tokens', type=int, default=2048,
                   help='maximum provider output tokens per response; recorded in the pool manifest')
    p.add_argument('--choice-contract', choices=['probabilities', 'label-confidence'], default='probabilities',
                   help='choice response shape; score/noul cases retain the probability contract')
    p.add_argument('--response-format', choices=['json_schema', 'json_object'], default='json_schema',
                   help='declared provider wire format; responses always use the same strict local validation')
    p.add_argument('--choice-enum-policy', choices=['include', 'omit'], default='include',
                   help='provider-side JSON-schema choice enum; local option validation remains strict')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--workers', type=int, default=4, help='total concurrent requests across paced quota groups')
    p.add_argument('--max-inflight-per-quota-group', type=int, default=2,
                   help='concurrent calls per quota group; request-start interval still applies')
    p.add_argument('--wait', action='store_true', help='wait for cooldowns instead of exiting with unfinished cases')
    p.add_argument('--wait-for-owner', action='store_true', help='queue this source behind the current project pool owner')
    args = p.parse_args()
    if (args.limit < 0 or args.timeout <= 0 or args.workers < 1 or
            args.max_inflight_per_quota_group < 1 or args.max_output_tokens < 1):
        p.error('limit must be nonnegative; timeout and max-output-tokens must be positive')
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
    state_bytes = state_path.read_bytes() if state_path.exists() else None
    state = json.loads(state_bytes) if state_bytes is not None else {'groups': {}, 'cursor': 0}
    groups = state.setdefault('groups', {})
    # Earlier runners could leave a daily-quota group disabled indefinitely when
    # Google supplied no retry delay. Migrate that exact legacy state only while
    # holding the shared quota lock; preserve provenance and wait before probing.
    migrated_daily_groups = []
    migration_now = time.time()
    state_sha_before_migration = hashlib.sha256(state_bytes).hexdigest() if state_bytes is not None else None
    for group_name, group in groups.items():
        if group.get('disabled') != DAILY_QUOTA_NO_RETRY_REASON:
            continue
        previous_disabled = group.pop('disabled')
        prior_not_before = group.get('not_before')
        scheduled = max(float(prior_not_before or 0), migration_now + DAILY_QUOTA_REPROBE_BASE_SECONDS)
        record = {
            'schema': 'natlang.google-daily-quota-reprobe/1',
            'reason': previous_disabled,
            'probe_count': 0,
            'next_probe_at': scheduled,
            'history': [],
        }
        _append_daily_reprobe_event(record, {
            'event': 'migrated_legacy_disabled_state',
            'at': migration_now,
            'source_state_sha256': state_sha_before_migration,
            'prior_not_before': prior_not_before,
            'quota_ids': group.get('quota_ids', []),
            'note': 'Migrated under the exclusive project-owner lock; first probe is delayed.'
        })
        group['daily_quota_reprobe'] = record
        group['not_before'] = scheduled
        migrated_daily_groups.append(group_name)
    if migrated_daily_groups:
        write_json(state_path, state)
    identity = {'schema': 'natlang.gemini-decision-pool/1', 'training_admission': False,
                'cases_sha256': hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),
                'choice_contract': args.choice_contract,
                'response_format': args.response_format,
                'choice_enum_policy': args.choice_enum_policy,
                'max_output_tokens': args.max_output_tokens,
                'models': models, 'endpoint': ENDPOINT, 'workers': args.workers,
                'max_inflight_per_quota_group': args.max_inflight_per_quota_group,
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
            reasoning_effort=model['reasoning_effort'], max_output_tokens=args.max_output_tokens,
            response_format=args.response_format, choice_contract=args.choice_contract,
            choice_enum_policy=args.choice_enum_policy)
    with ThreadPoolExecutor(max_workers=args.workers) as executor, out_path.open('a') as out, open(str(out_path) + '.attempts.jsonl', 'a') as journal:
        while cases or pending:
            now = time.time()
            busy = Counter(model['quota_group'] for case, model in pending.values())
            order = [(state.get('cursor', 0) + i) % len(models) for i in range(len(models))]
            for index in order:
                if not cases or len(pending) >= args.workers:
                    break
                model = models[index]
                group = groups.setdefault(model['quota_group'], {})
                daily_reprobe = group.get('daily_quota_reprobe')
                per_group_limit = 1 if daily_reprobe else args.max_inflight_per_quota_group
                if (busy[model['quota_group']] >= args.max_inflight_per_quota_group or
                        group.get('disabled') or group.get('not_before', 0) > now):
                    continue
                if busy[model['quota_group']] >= per_group_limit:
                    continue
                case = cases.popleft()
                state['cursor'] = (index + 1) % len(models)
                if daily_reprobe:
                    # Persist a safe retry time with the probe start, before the
                    # request is submitted. If the owner dies before recording its
                    # response, the next owner will not immediately repeat the call.
                    fallback_delay = min(
                        DAILY_QUOTA_REPROBE_BASE_SECONDS * (2 ** min(daily_reprobe.get('probe_count', 0), 5)),
                        DAILY_QUOTA_REPROBE_MAX_SECONDS,
                    )
                    group['not_before'] = now + max(model['interval_seconds'], fallback_delay)
                    daily_reprobe['fallback_retry_at'] = group['not_before']
                    daily_reprobe['last_probe_started_at'] = now
                    daily_reprobe['last_probe_model'] = model['id']
                    _append_daily_reprobe_event(daily_reprobe, {
                        'event': 'timed_probe_started', 'at': now, 'model': model['id'],
                        'attempt_number': daily_reprobe.get('probe_count', 0) + 1,
                        'fallback_retry_at': daily_reprobe['fallback_retry_at'],
                    })
                else:
                    group['not_before'] = now + model['interval_seconds']
                write_json(state_path, state)
                pending[executor.submit(ask, case, model)] = (case, model)
                busy[model['quota_group']] += 1
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
                    # The persisted delay is the maximum parsed server Retry-After /
                    # Google RetryInfo value. Preserve an explicit zero as a supplied
                    # delay instead of treating it as missing metadata.
                    delays = [float(h['retry_after_seconds']) for h in provider['retry_history']
                              if 'retry_after_seconds' in h]
                    has_server_delay = bool(delays)
                    quota_ids = [q for h in provider['retry_history'] for q in h.get('quota_ids', [])]
                    per_day_quota = any('PerDay' in q for q in quota_ids)
                    generic_backoff = min(300, 15 * 2 ** min(group['failure_streak'] - 1, 5))
                    # A positive Retry-After / RetryInfo is a lower bound; retain
                    # local pacing even when a server explicitly reports zero.
                    delay = max(max(delays, default=0), generic_backoff)
                    if per_day_quota and not has_server_delay:
                        reprobe = group.get('daily_quota_reprobe')
                        if reprobe is None:
                            reprobe = {
                                'schema': 'natlang.google-daily-quota-reprobe/1',
                                'reason': DAILY_QUOTA_NO_RETRY_REASON,
                                'probe_count': 0,
                                'history': [],
                            }
                            group['daily_quota_reprobe'] = reprobe
                            reprobe_delay = DAILY_QUOTA_REPROBE_BASE_SECONDS
                        else:
                            if reprobe.get('last_probe_started_at') is not None:
                                reprobe['probe_count'] = reprobe.get('probe_count', 0) + 1
                            reprobe_delay = min(
                                DAILY_QUOTA_REPROBE_BASE_SECONDS * (2 ** min(reprobe.get('probe_count', 0), 5)),
                                DAILY_QUOTA_REPROBE_MAX_SECONDS,
                            )
                        observed_at = time.time()
                        last_observed = {
                            'event': 'daily_quota_without_retry_delay',
                            'at': observed_at,
                            'status': status,
                            'answer_error': answer.get('error'),
                            'quota_ids': quota_ids,
                            'response_sha256': provider.get('response_sha256'),
                            'server_delay_present': False,
                            'probe_count': reprobe.get('probe_count', 0),
                            'scheduled_delay_seconds': reprobe_delay,
                        }
                        reprobe.pop('last_probe_started_at', None)
                        reprobe.pop('last_probe_model', None)
                        _append_daily_reprobe_event(reprobe, last_observed)
                        reprobe['next_probe_at'] = observed_at + reprobe_delay
                        delay = reprobe_delay
                    elif per_day_quota and has_server_delay:
                        reprobe = group.get('daily_quota_reprobe')
                        if reprobe is not None:
                            _append_daily_reprobe_event(reprobe, {
                                'event': 'server_retry_delay_honored', 'at': time.time(),
                                'status': status, 'quota_ids': quota_ids,
                                'response_sha256': provider.get('response_sha256'),
                                'retry_after_seconds': max(delays),
                            })
                            reprobe.pop('last_probe_started_at', None)
                            reprobe.pop('last_probe_model', None)
                            reprobe['next_probe_at'] = time.time() + delay
                    group.update(not_before=time.time() + delay, quota_ids=quota_ids)
                    cases.appendleft(case)
                    print(json.dumps({'event': 'model_cooldown', 'model': model['id'], 'seconds': delay}), flush=True)
                elif status in (401, 403, 404):
                    group['disabled'] = f'http_{status}_requires_review'
                    cases.appendleft(case)
                    print(json.dumps({'event': 'model_disabled', 'model': model['id'], 'status': status}), flush=True)
                else:
                    # HTTP 400 describes this request's argument/payload, not the
                    # model quota group's health. Preserve the failed case as a
                    # provider error row below and continue using the model for
                    # later cases; any alternate wire format needs a fresh,
                    # explicitly reviewed plan rather than a hidden fallback.
                    if status == 400:
                        provider['failure_class'] = 'request_specific_http_400'
                        structured_error = next((h.get('provider_error') for h in provider.get('retry_history', [])
                                                 if h.get('provider_error')), {})
                        print(json.dumps({'event': 'provider_request_rejected',
                                          'id': case['id'], 'model': model['id'],
                                          'status': status,
                                          'provider_error_code': structured_error.get('code'),
                                          'provider_error_status': structured_error.get('status'),
                                          'response_sha256': provider.get('response_sha256')}), flush=True)
                    group['failure_streak'] = 0
                    reprobe = group.get('daily_quota_reprobe')
                    if reprobe is not None:
                        # A successful response resolves the probe hold; return to
                        # ordinary request pacing rather than retaining its crash guard.
                        group['not_before'] = time.time() + model['interval_seconds']
                        _finish_daily_reprobe(group, reprobe, {
                            'event': 'timed_probe_http_succeeded', 'at': time.time(),
                            'status': status, 'model': model['id'],
                            'response_sha256': provider.get('response_sha256'),
                        })
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
