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
Several teachers share one WORK with disjoint `--shard K N`.

  pause:   touch WORK/PAUSE   (running episodes finish, no new ones start; remove the file to continue)
  stop:    systemctl --user stop <unit>   (killed episodes are redone on the next launch)
  resume:  launch the same command again (finished tasks are skipped)
"""
import argparse
import json
import os
import random
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

    def _take(self):
        with self.lock:
            while self.cursor < min(len(self.tasks), self.args.limit or len(self.tasks)):
                index = self.cursor
                self.cursor += 1
                if index not in self.done and index % self.args.shard[1] == self.args.shard[0]:
                    return index
            return None

    def run_task(self, index):
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
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens)
            key = entry['key']
            queue_file = directory / f'{key}.queue.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file),
                       str(directory / f'{key}.journal.jsonl'), '--runtime', str(REPO / 'ts-host'),
                       '--server', a.endpoint, '--chat-completions-url', a.chat_completions_url or a.endpoint + '/v1/chat/completions',
                       '--model-id', a.model_id, '--model-concurrency', '1',
                       '--chat-request-config', str(self.work / 'chat-config.json')]
            if a.api_key_env:
                command += ['--api-key-env', a.api_key_env]
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
                  'teacher': {'model_id': a.model_id, 'endpoint': a.endpoint},
                  'seconds': round(time.time() - started, 1),
                  'episodes': [{'sample': r['sample'], 'outcome': r['outcome'],
                                'reward': (r.get('reward') or {}).get('quality'),
                                'graded': (r.get('reward') or {}).get('graded'), 'turns': len(r['turns']),
                                'completion_tokens': sum((t.get('usage') or {}).get('completion_tokens') or 0
                                                         for t in r['turns'])} for r in rollouts],
                  'rollouts': str(directory / 'rollouts.jsonl')}
        with self.lock:
            with self.manifest.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            self.done.add(index)
        print(json.dumps({'event': 'task scored', **{k: record[k] for k in ('index', 'task_id', 'seconds', 'episodes')}}),
              flush=True)
        return report

    def slot(self):
        while not self.stop.is_set():
            if (self.work / 'PAUSE').exists():
                self.stop.wait(10)
                continue
            index = self._take()
            if index is None:
                return
            try:
                self.run_task(index)
            except Exception as error:   # one task's failure never stops the generation
                print(json.dumps({'event': 'task failed', 'index': index, 'error': repr(error)[:400]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work', required=True)
    p.add_argument('--endpoint', required=True, help='teacher server base URL (OpenAI-compatible)')
    p.add_argument('--chat-completions-url', help='exact completion URL (default ENDPOINT/v1/chat/completions)')
    p.add_argument('--model-id', required=True)
    p.add_argument('--api-key-env', help='environment variable holding the API key (name only)')
    p.add_argument('--chat-config', default='{"top_p": 0.95}', help='JSON request controls passed unchanged')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--slots', type=int, default=2)
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
    chat_config = json.loads(args.chat_config)
    (work / 'chat-config.json').write_text(json.dumps(chat_config) + '\n')
    with (work / 'launches.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv,
                                 'code': code_provenance(), 'chat_config': chat_config,
                                 'teacher': {'model_id': args.model_id, 'endpoint': args.endpoint}}) + '\n')
    generator = Generator(args)

    def stop(signum, frame):
        generator.stop.set()
    signal.signal(signal.SIGTERM, stop)
    threads = [threading.Thread(target=generator.slot, daemon=True) for _ in range(args.slots)]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(5)


if __name__ == '__main__':
    main()
