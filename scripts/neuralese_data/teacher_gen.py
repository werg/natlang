#!/usr/bin/env python3
"""Teacher episodes on the GRPO task pool, for distillation (SFT) into the LFM3 actor.

Any OpenAI-compatible chat endpoint is the teacher (a local llama-server, a hosted free tier, a paid API): `--endpoint`
and `--model-id` (plus `--api-key-env` for hosted ones). Episodes run through the GRPO path itself: `task_rollouts.prepare`
(student root, full prompt rendering, the shared `EPISODE_MODEL_CALLS` budget), `run_bonsai_queue.py` (the frozen
collector, without the RL capture: the teacher's tokens are not the student's) and `task_rollouts.score` (the task's exact
oracle). Each scored episode's `natlang.rollout/1` record (turns = {messages, tools, target}) is the SFT material; the
successes are the distillation set, failures keep their scores.

Tasks are the GRPO pools minus the held closure (`grpo.load_tasks`: held-v3 `exclusion_ids` incl. held-v2, shared source
groups, `ops/pool-errata`, the missing ext_codebase repos), ordered once into `WORK/tasks.jsonl`: tasks in the base's pass band
first, then tasks the base fails, then unmeasured ones, then those it solves, each tier round-robin over families.
Several teachers share one WORK: each claims a task before running it (WORK/claims), so faster teachers take more.

  pause:   touch WORK/PAUSE   (running episodes finish, no new ones start; remove the file to continue);
           --pause-below-available-gb N does the same while host MemAvailable is below N GB
  stop:    systemctl --user stop <unit>   (killed episodes are redone on the next launch)
  resume:  launch the same command again (finished tasks are skipped)
  provider failures (an episode ending before its first model turn) are no verdict: deferred.jsonl, retried twice
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common.paths import resolve_str  # noqa: E402

BASE_RATES = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'exp1-rft', 'base-pass-rates.json')


def order_tasks(rows, rates, band, seed):
    """Tiers by the base's outcome (in-band first, then base failures, then unmeasured, then base successes), each tier
    round-robin over a shuffled family order: no family repeats within a tier before every family had its turn."""
    rng = random.Random(seed)
    low, high = band

    def rank(row):
        rate = rates.get(row['id'])
        if not rate or not rate.get('scored'):
            return 2
        p = rate['successes'] / rate['scored']
        return 0 if low <= p <= high else 1 if p < low else 3

    families = {}
    for row in rows:
        families.setdefault(row.get('family') or '?', []).append(row)
    names = sorted(families)
    rng.shuffle(names)
    keyed = []
    for order, name in enumerate(names):
        members = families[name]
        rng.shuffle(members)
        seen = {}
        for row in members:   # the k-th task of its tier in this family comes in the tier's k-th family round
            tier = rank(row)
            seen[tier] = seen.get(tier, -1) + 1
            keyed.append(((tier, seen[tier], order), row))
    keyed.sort(key=lambda item: item[0])
    ordered = [row for _, row in keyed]
    return ordered, rank


GEMINI_HOST_CALL_FIELDS = {'extra_content': {'google': {'thought_signature': 'skip_thought_signature_validator'}}}


def default_chat_config(url):
    """Request controls when none are given: top-p; for Gemini's OpenAI endpoint also no `seed` field (it rejects it) and
    Google's documented bypass signature on the function calls the runtime wrote itself (it rejects unsigned ones)."""
    gemini = 'generativelanguage.googleapis.com' in url
    return {'top_p': 0.95, **({'x_natlang_omit_seed': True, 'x_natlang_host_call_fields': GEMINI_HOST_CALL_FIELDS} if gemini else {})}


def mem_available_gb():
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return float('inf')


def completions_url(endpoint):
    endpoint = endpoint.rstrip('/')
    return endpoint + ('/chat/completions' if endpoint.endswith('/v1') else '/v1/chat/completions')


def chat_config_path(work, model_id):
    """Each teacher sharing WORK keeps its own request controls."""
    return Path(work) / f"chat-config.{re.sub(r'[^A-Za-z0-9._-]+', '_', model_id)}.json"


# the collector's message when a provider kept failing a request (429, 5xx, transport) through every retry
PROVIDER_FAILURE = 'provider request retry exhausted'
# decide() collection needs the actor's exact native capture, which a hosted teacher endpoint does not give
UNSUPPORTED_DECIDE = 'decision collection needs its exact native capture'


class Generator:
    def __init__(self, args):
        self.args, self.work = args, Path(args.work)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.env = {**os.environ, 'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}
        from natlang_neuralese.rl.held_eval import series
        self.env['NATLANG_MODEL_FACING'] = series()['model_facing']
        self.tasks = [json.loads(line) for line in (self.work / 'tasks.jsonl').open()]
        self.cursor = 0
        self.manifest = self.work / 'manifest.jsonl'
        self.done = set()
        if self.manifest.exists():
            self.done = {json.loads(line)['index'] for line in self.manifest.open() if line.strip()}
        (self.work / 'claims').mkdir(exist_ok=True)
        self.retry = []           # tasks whose episodes the provider failed (no verdict), retried after the cursor
        self.attempts = {}

    def _no_verdict(self, record):
        """Every episode ended unresolved before its first model turn or as an infrastructure error from exhausted provider
        retries (e.g. a 429 mid-episode; a model-call budget stop is the teacher's own verdict): the provider failed (rate limit, outage), not the teacher. Such a record is no teacher verdict; it goes to deferred.jsonl and the task is retried."""
        return 'unsupported' not in record and all(
            (e['outcome'] == 'infrastructure_error' and record.get('provider_failure')) or
            (e['outcome'] == 'unresolved' and e['turns'] == 0)
            for e in record['episodes'])

    def _claim(self, index, teacher):
        """Several teachers share one WORK and steal tasks from one ordered list: WORK/claims/tINDEX names the process
        running a task (state running) or the one that recorded it (state done). A dead process's claim is taken over."""
        path = self.work / 'claims' / f't{index:06d}'
        mine = json.dumps({'pid': os.getpid(), 'teacher': teacher.model_id, 'state': 'running'})
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            try:
                held = json.loads(path.read_text() or '{}')
            except (OSError, ValueError):
                return False   # being written by its claimant
            if held.get('state') == 'done':
                self.done.add(index)
                return False
            try:
                os.kill(int(held.get('pid', 0)), 0)
                return False   # its claimant is alive
            except (ProcessLookupError, ValueError):
                pass
            except PermissionError:
                return False
            tmp = path.with_suffix(f'.{os.getpid()}')
            tmp.write_text(mine)
            os.replace(tmp, path)
            time.sleep(0.2)
            return json.loads(path.read_text() or '{}').get('pid') == os.getpid()
        with os.fdopen(fd, 'w') as handle:
            handle.write(mine)
        return True

    def _release(self, index, state, teacher):
        path = self.work / 'claims' / f't{index:06d}'
        if state == 'done':
            tmp = path.with_suffix(f'.{os.getpid()}')
            tmp.write_text(json.dumps({'pid': os.getpid(), 'teacher': teacher.model_id, 'state': 'done'}))
            os.replace(tmp, path)
        else:
            path.unlink(missing_ok=True)

    def _take(self, teacher):
        while time.time() < teacher.cool_until and not self.stop.is_set():
            self.stop.wait(5)
        with self.lock:
            while self.retry and (self.retry[0][0] <= time.time() or self.cursor >= len(self.tasks)):
                index = self.retry.pop(0)[1]
                if index not in self.done and self._claim(index, teacher):
                    return index
            while self.cursor < min(len(self.tasks), self.args.limit or len(self.tasks)):
                index = self.cursor
                self.cursor += 1
                if index not in self.done and index % self.args.shard[1] == self.args.shard[0] and self._claim(index, teacher):
                    return index
            return None

    def run_task(self, index, teacher):
        from natlang_neuralese.common.jsonio import read_jsonl
        from natlang_neuralese.rl import task_rollouts
        a, task = self.args, self.tasks[index]
        row = task['row']
        directory = self.work / 'episodes' / f't{index:06d}'
        if directory.exists():   # an unscored attempt (stopped mid-episode) is redone whole
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        (directory / 'cases.jsonl').write_text(json.dumps(row) + '\n')
        task_rollouts.prepare(SimpleNamespace(
            cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=max(2, a.samples), temperature=a.temperature,
            case_seconds=a.case_seconds, max_model_requests=a.max_model_requests, max_turns=a.max_turns,
            seed_base=a.seed_base + 100 * index, root_mode='student', match_seeds_from=None,
            policy_checkpoint_sha256=None))
        # prepare builds GRPO groups (>= 2 samples); a teacher may want one: the queue keeps the first `samples`
        entries = list(read_jsonl(directory / 'queue.jsonl'))[:a.samples]
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
        started = time.time()
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens, request_retries=a.request_retries)
            key = entry['key']
            queue_file = directory / f'{key}.queue.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file),
                       str(directory / f'{key}.journal.jsonl'), '--runtime', str(REPO / 'ts-host'),
                       '--server', teacher.endpoint, '--chat-completions-url', teacher.chat_completions_url,
                       '--model-id', teacher.model_id, '--model-concurrency', '1',
                       '--chat-request-config', str(chat_config_path(self.work, teacher.model_id))]
            if teacher.api_key_env:
                command += ['--api-key-env', teacher.api_key_env]
            with (directory / f'{key}.log').open('a') as handle:
                process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while process.poll() is None:
                    if self.stop.wait(2):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        return
            if process.returncode < 0:
                return   # killed: no verdict; redone on the next launch
        with (directory / 'journal.jsonl').open('w') as merged:
            for path in sorted(directory.glob('*.journal.jsonl')):
                if path.name != 'journal.jsonl':
                    merged.write(path.read_text())
        report = task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        rollouts = list(read_jsonl(directory / 'rollouts.jsonl'))
        record = {'index': index, 'task_id': row['id'], 'family': row.get('family'), 'base_rank': task['base_rank'],
                  'teacher': {'model_id': teacher.model_id, 'endpoint': teacher.endpoint},
                  'seconds': round(time.time() - started, 1),
                  'episodes': [{'sample': r['sample'], 'outcome': r['outcome'],
                                'reward': (r.get('reward') or {}).get('quality'),
                                'graded': (r.get('reward') or {}).get('graded'), 'turns': len(r['turns']),
                                'completion_tokens': sum((t.get('usage') or {}).get('completion_tokens') or 0
                                                         for t in r['turns'])} for r in rollouts],
                  'rollouts': str(directory / 'rollouts.jsonl')}
        logs = ''.join(path.read_text(errors='replace') for path in directory.glob('*.log'))
        record['provider_failure'] = PROVIDER_FAILURE in logs
        if UNSUPPORTED_DECIDE in logs:   # a deterministic harness limit, not a provider failure: recorded, never retried
            record['unsupported'] = 'decide-native-capture'
        with self.lock:
            if self._no_verdict(record):
                self.attempts[index] = self.attempts.get(index, 0) + 1
                with (self.work / 'deferred.jsonl').open('a') as handle:
                    handle.write(json.dumps({**record, 'attempt': self.attempts[index]}) + '\n')
                teacher.cool_until = max(teacher.cool_until, time.time() + self.args.provider_cooldown)
                if self.attempts[index] < 3:
                    self.retry.append((time.time() + 600, index))
                self._release(index, 'deferred', teacher)
                print(json.dumps({'event': 'task deferred', 'index': index, 'attempt': self.attempts[index]}), flush=True)
                return report
            with self.manifest.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            self.done.add(index)
            self._release(index, 'done', teacher)
        print(json.dumps({'event': 'task scored', **{k: record[k] for k in ('index', 'task_id', 'seconds', 'episodes')}}),
              flush=True)
        return report

    def slot(self, teacher):
        while not self.stop.is_set():
            if (self.work / 'PAUSE').exists() or mem_available_gb() < self.args.pause_below_available_gb:
                self.stop.wait(10)
                continue
            index = self._take(teacher)
            if index is None:
                return
            try:
                self.run_task(index, teacher)
            except Exception as error:   # one task's failure never stops the generation
                self._release(index, 'failed', teacher)
                print(json.dumps({'event': 'task failed', 'index': index, 'error': repr(error)[:400]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work', required=True)
    p.add_argument('--teacher', action='append', default=[], type=json.loads, metavar='JSON',
                   help='one teacher of several in this process (the task list is loaded once): {"model_id", "endpoint", '
                        '"chat_completions_url", "api_key_env", "slots", "chat_config"}; repeatable. Without it the '
                        'single-teacher flags below describe the one teacher')
    p.add_argument('--endpoint', help='teacher server base URL (OpenAI-compatible)')
    p.add_argument('--chat-completions-url', help='exact completion URL (default ENDPOINT/v1/chat/completions)')
    p.add_argument('--model-id')
    p.add_argument('--api-key-env', help='environment variable holding the API key (name only)')
    p.add_argument('--chat-config', type=json.loads, help='JSON request controls passed unchanged (default: top_p 0.95)')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--slots', type=int, default=2)
    p.add_argument('--request-retries', type=int, default=8,
                   help='same-request retries after a provider failure (429, 5xx), with backoff honouring Retry-After')
    p.add_argument('--provider-cooldown', type=float, default=120,
                   help='seconds all of a teacher\'s slots wait before taking a new task after an episode the provider failed')
    p.add_argument('--pause-below-available-gb', type=float, default=0,
                   help='take no new task while MemAvailable is below this (a co-located learner phase needs the memory)')
    p.add_argument('--samples', type=int, default=1, help='episodes per task')
    p.add_argument('--limit', type=int, default=0, help='stop after the first N ordered tasks (0: all)')
    p.add_argument('--shard', type=int, nargs=2, default=(0, 1), metavar=('K', 'N'),
                   help='take only tasks with index %% N == K (several teachers share one ordered list)')
    p.add_argument('--context-tokens', type=int, default=65536)
    p.add_argument('--case-seconds', type=int, default=3600)
    p.add_argument('--max-model-requests', type=int, default=None, help='default: EPISODE_MODEL_CALLS (40)')
    p.add_argument('--max-turns', type=int, default=128)
    p.add_argument('--pass-band', type=float, nargs=2, default=(0.1, 0.9))
    p.add_argument('--base-rates', default=BASE_RATES)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--seed-base', type=int, default=50_000_000)
    args = p.parse_args()
    from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS
    args.max_model_requests = args.max_model_requests or EPISODE_MODEL_CALLS
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    from natlang_neuralese.common.code_provenance import code_provenance
    if not (work / 'tasks.jsonl').exists():
        import grpo
        pool_args = SimpleNamespace(pool=grpo.POOLS, held_manifest=grpo.HELD, exclude_manifest=[], task_ids=None,
                                    exclude_family_regex=r'^ext_codebase-(py|natlang)_codebase_', exclude_ids=[],
                                    no_default_errata=False, seed=args.seed, pass_rates=args.base_rates,
                                    pass_band=args.pass_band, pass_min_scored=1, explore=None,
                                    max_model_requests=args.max_model_requests, case_seconds=args.case_seconds)
        rows, excluded = grpo.load_tasks(pool_args, work)   # also writes WORK/pool.json (pool shas, held closure)
        rates = json.loads(Path(args.base_rates).read_text()) if args.base_rates else {}
        ordered, rank = order_tasks(rows, rates, args.pass_band, args.seed)
        with (work / 'tasks.jsonl.tmp').open('w') as handle:
            for row in ordered:
                handle.write(json.dumps({'row': row, 'base_rank': ['band', 'base_fails', 'unmeasured', 'base_solves'][rank(row)]}) + '\n')
        os.replace(work / 'tasks.jsonl.tmp', work / 'tasks.jsonl')
    specs = args.teacher or [{'model_id': args.model_id, 'endpoint': args.endpoint, 'api_key_env': args.api_key_env,
                              'chat_completions_url': args.chat_completions_url, 'slots': args.slots,
                              'chat_config': args.chat_config}]
    teachers = []
    for spec in specs:
        if not spec.get('model_id') or not spec.get('endpoint'):
            p.error('each teacher needs model_id and endpoint')
        url = spec.get('chat_completions_url') or completions_url(spec['endpoint'])
        teacher = SimpleNamespace(model_id=spec['model_id'], endpoint=spec['endpoint'].rstrip('/'), chat_completions_url=url,
                                  api_key_env=spec.get('api_key_env'), slots=int(spec.get('slots', 2)),
                                  chat_config=default_chat_config(url) if spec.get('chat_config') is None else spec['chat_config'],
                                  cool_until=0.)
        chat_config_path(work, teacher.model_id).write_text(json.dumps(teacher.chat_config) + '\n')
        teachers.append(teacher)
    with (work / 'launches.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv,
                                 'code': code_provenance(),
                                 'teachers': [{k: v for k, v in vars(t).items() if k != 'cool_until'} for t in teachers]}) + '\n')
    generator = Generator(args)

    def stop(signum, frame):
        generator.stop.set()
    signal.signal(signal.SIGTERM, stop)
    threads = [threading.Thread(target=generator.slot, args=(teacher,), daemon=True)
               for teacher in teachers for _ in range(teacher.slots)]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(5)


if __name__ == '__main__':
    main()
