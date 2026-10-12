"""In-run task-success evaluation (owner decision 2026-10-12, plans/neuralese/DECISIONS.md "Qualification is task success
plus drift").

A fixed subset of the protected held task packet is executed for real at the training eval cadence, against the
model being trained, in the trainer's own process: the stage's serve ``Engine`` (the object direct SFT trains
through) is served on a loopback port for the duration of the evaluation, and the shared TS collector
(``scripts/run_bonsai_queue.py`` -> ``ts-host/dist/teacher/cli.js``) executes each task's authored root program,
asks the model for every child decision, and runs the tool calls (eval, return_result, ...) in the native runtime.
Scoring is the shared exact scorer (``eval.task_packet.score_rows``): a task succeeds only when the collector's own
host checks accept it (``outcome.accepted``); expected values stay evaluator-side.

Crisp by design: this is an exact verifier of what the stage produces (task outcomes), not a policy. No extra model
memory is used beyond the KV cache of the served requests.

The spec (stage parameter ``task_eval``, JSON):
  packet          {"path", "sha256"}: a registered held task packet (natlang.program records; rows with an
                  authored_root_eval contract run their authored root, all others run the whole program)
  cases           packet row indices of the fixed subset (default: every third row, 16 of 48)
  prompt_rendering  serving prompt policy; default: the policy the stage's training rows declare
  case_seconds    wall budget per task (collection resource limit, not a model label; default 600)
  max_model_requests / max_turns   per-task request and turn budgets (default 48 / 60)
  workers         tasks executed concurrently (default 4; the engine batches their requests)
  min_successes   absolute floor on the final evaluation's task successes (default 0)
  reference_successes  the line's base model's successes on this subset under its native rendering
  noise_successes  declared run-to-run noise of the success count on this subset (from repeat runs)
  noise_repeats   runs of the stage's first evaluation (default 2): their spread is the measured noise; the bar
                  uses the larger of the declared and the measured noise

The pass bar is relative (coordinator 2026-10-12): the final evaluation is eligible, reaches the base
(``reference_successes``; an unmeasured reference fails the gate) and ``min_successes``, and is not below the
previous eligible evaluation of the stage minus ``noise_successes``. Every evaluation records the same comparison
(``regressed``) for monitoring; only the final one gates.
  runtime         TS runtime directory with dist/teacher (default: the live checkout's ts-host, the newest code)
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CASES = tuple(range(0, 48, 3))
SCHEMA = 'natlang.in-run-task-eval/1'


def parse_task_eval(value):
    """The ``--task-eval`` JSON (or None). Validates the declared shape; the packet hash is checked at first use."""
    if value in (None, '', 'null'):
        return None
    spec = json.loads(value) if isinstance(value, str) else dict(value)
    packet = spec.get('packet') or {}
    if not isinstance(packet.get('path'), str) or not isinstance(packet.get('sha256'), str):
        raise ValueError('task_eval.packet needs {"path", "sha256"}')
    cases = spec.setdefault('cases', list(DEFAULT_CASES))
    if (not isinstance(cases, list) or not cases or len(set(cases)) != len(cases) or
            any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in cases)):
        raise ValueError('task_eval.cases must be distinct nonnegative row indices')
    for key, default in (('case_seconds', 600), ('max_model_requests', 48), ('max_turns', 60), ('workers', 4),
                         ('min_successes', 0), ('noise_successes', 0), ('noise_repeats', 2)):
        spec.setdefault(key, default)
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < (
                0 if key in ('min_successes', 'noise_successes') else 1):
            raise ValueError(f'task_eval.{key} must be a positive integer')
    reference = spec.setdefault('reference_successes', None)
    if reference is not None and (isinstance(reference, bool) or not isinstance(reference, int) or reference < 0):
        raise ValueError('task_eval.reference_successes must be a nonnegative integer or null (unmeasured)')
    if spec.get('prompt_rendering') not in (None, 'full', 'program-header'):
        raise ValueError('task_eval.prompt_rendering must be full or program-header')
    return spec


def training_prompt_rendering(rows) -> str:
    """The serving policy the stage's training rows declare (undeclared rows are the historical full prompt)."""
    counts = collections.Counter((row.get('prompt_rendering') or 'full') for row in rows)
    return counts.most_common(1)[0][0] if counts else 'full'


def node_path_env() -> dict:
    """Environment with a ``node`` on PATH (training containers mount the host's /home but ship no node)."""
    env = dict(os.environ)
    if shutil.which('node', path=env.get('PATH')):
        return env
    candidates = [os.environ.get('NATLANG_NODE_DIR'), str(Path.home() / '.local' / 'bin')]
    candidates += sorted(str(p / 'bin') for p in (Path.home() / '.local' / 'share' / 'natlang-dev').glob('node-*'))
    for directory in filter(None, candidates):
        if (Path(directory) / 'node').exists():
            env['PATH'] = directory + os.pathsep + env.get('PATH', '')
            return env
    raise RuntimeError('task_eval needs node on PATH (or NATLANG_NODE_DIR) to run the TS collector')


def _git_head() -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class TaskEval:
    """Serve ``engine`` on loopback and execute the fixed held subset against it."""

    def __init__(self, spec: dict, out: Path, engine, *, prompt_rendering: str):
        from ..common.hashing import sha256_file_hex
        from .task_packet import read_jsonl

        self.spec, self.out, self.engine = spec, Path(out), engine
        self.prompt_rendering = spec.get('prompt_rendering') or prompt_rendering
        from ..common.paths import root as logical_root
        # Repo-relative paths resolve against the live checkout (a training run executes a code snapshot that
        # carries neither runs/ nor the built ts-host).
        self.repo = logical_root('repo')
        packet = Path(spec['packet']['path'])
        if not packet.is_absolute():
            packet = self.repo / packet
        if sha256_file_hex(packet) != spec['packet']['sha256']:
            raise ValueError(f'task_eval packet changed: {packet}')
        rows = read_jsonl(packet)
        if max(spec['cases']) >= len(rows):
            raise ValueError('task_eval.cases index past the packet end')
        self.rows = [rows[i] for i in spec['cases']]
        for row in self.rows:
            if row.get('split') != 'test':
                raise ValueError(f'task_eval row is not a held (test) row: {row.get("id")}')
        self.root = self.out / 'task-eval'
        self.root.mkdir(parents=True, exist_ok=True)
        self.subset = self.root / 'subset.jsonl'
        text = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in self.rows)
        if not self.subset.exists() or self.subset.read_text(encoding='utf-8') != text:
            self.subset.write_text(text, encoding='utf-8')

    def previous(self, step: int) -> dict | None:
        """The stage's last eligible evaluation before ``step`` on the same packet and cases (None if none)."""
        log = self.out / 'task-eval.jsonl'
        if not log.is_file():
            return None
        found = None
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (row.get('eligible') and row.get('step', step) < step and
                    row.get('packet_sha256') == self.spec['packet']['sha256'] and
                    row.get('case_indices') == self.spec['cases']):
                found = row
        return found

    def _queues(self, run: Path) -> list[tuple[Path, Path, list[dict]]]:
        workers = min(self.spec['workers'], len(self.rows))
        queues = []
        for worker in range(workers):
            directory = run / f'worker-{worker:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            entries = []
            for index in range(worker, len(self.rows), workers):
                case = directory / f'case-{index:04d}'
                entries.append({'index': index, 'count': 1, 'key': f'task-{index:04d}',
                                'source': str(self.subset), 'jobs': str(case / 'jobs'),
                                'log': str(case / 'collector.log'), 'output': str(case / 'results.jsonl'),
                                'case_seconds': self.spec['case_seconds'], 'context_tokens': 32768,
                                'max_model_requests': self.spec['max_model_requests'],
                                'max_turns': self.spec['max_turns'], 'request_retries': 1,
                                'transport_retries': 0, 'seed': 400000 + self.spec['cases'][index],
                                'text_neuralese_emulation': False})
            queue = directory / 'queue.jsonl'
            queue.write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries), encoding='utf-8')
            queues.append((queue, directory / 'journal.jsonl', entries))
        return queues

    def _execute(self, run: Path) -> tuple[dict, list[int]]:
        """Serve the engine and run every subset task once into ``run``; the exact score and collector exit codes."""
        from ..serve.http import serve
        from .task_packet import read_jsonl, score_rows

        if run.exists():
            shutil.rmtree(run)  # a rerun of the same step (resume) replaces an interrupted attempt
        run.mkdir(parents=True)
        engine, hf = self.engine, getattr(self.engine.backbone, 'hf', None)
        was_training = bool(hf is not None and hf.training)
        previous_mode = engine.prompt_rendering
        server = None
        try:
            if hf is not None:
                hf.eval()
            engine.prompt_rendering = self.prompt_rendering
            # Engine.stop() leaves its stop flag set; reset it so the training engine serves once per evaluation.
            # (Done here, not in serve/engine.py, whose bytes are part of the prompt-rendering identity.)
            import queue as queue_module
            engine._stopping, engine._incoming, engine._active = False, queue_module.Queue(), []
            engine.start()
            server = serve(engine, '127.0.0.1', 0)
            threading.Thread(target=server.serve_forever, name='task-eval-http', daemon=True).start()
            url = f'http://127.0.0.1:{server.server_address[1]}'
            env = node_path_env()
            children = []
            queues = self._queues(run)
            for queue, journal, _ in queues:
                log = (queue.parent / 'supervisor.log').open('w')
                children.append((subprocess.Popen(
                    [sys.executable, str(ROOT / 'scripts' / 'run_bonsai_queue.py'), str(queue), str(journal),
                     '--runtime', str(self.spec.get('runtime') or self.repo / 'ts-host'), '--server', url,
                     '--chat-completions-url', url + '/v1/chat/completions', '--model-concurrency', '2'],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            codes = []
            for child, log in children:
                codes.append(child.wait())
                log.close()
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            engine.stop()
            engine.prompt_rendering = previous_mode
            if hf is not None and was_training:
                hf.train()
        finishes, results, keys = {}, [], {}
        for queue, journal, entries in queues:
            if journal.is_file():
                for line in journal.read_text().splitlines():
                    if line.strip():
                        event = json.loads(line)
                        if event.get('event') == 'finish':
                            finishes[f'{queue.parent.name}/{event["key"]}'] = event
            for entry in entries:
                keys[entry['index']] = f'{queue.parent.name}/{entry["key"]}'
                if Path(entry['output']).is_file():
                    results.extend(read_jsonl(entry['output']))
        return score_rows(self.rows, results, finishes, keys), codes

    def measured_noise(self) -> int | None:
        """The largest success-count difference between repeat runs of one model recorded in this stage's log."""
        log = self.out / 'task-eval.jsonl'
        values = []
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('packet_sha256') == self.spec['packet']['sha256'] and row.get('repeat_noise') is not None:
                    values.append(row['repeat_noise'])
        return max(values) if values else None

    def run(self, step: int) -> dict:
        started = time.time()
        run = self.root / f'step-{step:07d}'
        score, codes = self._execute(run)
        eligible = score['eligible_for_best'] and all(code == 0 for code in codes)
        # Noise from repeat runs (coordinator 2026-10-12): the first evaluation of a stage runs the same model again
        # (noise_repeats total runs); the spread of the success counts is this stage's measured noise.
        repeats, repeat_noise = [], None
        if eligible and self.measured_noise() is None and self.spec.get('noise_repeats', 2) > 1:
            for index in range(1, self.spec.get('noise_repeats', 2)):
                again, again_codes = self._execute(run.with_name(run.name + f'-repeat{index}'))
                if again['eligible_for_best'] and all(code == 0 for code in again_codes):
                    repeats.append(again['task_successes'])
            if repeats:
                counts = [score['task_successes'], *repeats]
                repeat_noise = max(counts) - min(counts)
        measured = repeat_noise if repeat_noise is not None else self.measured_noise()
        noise = max(self.spec['noise_successes'], measured or 0)
        previous = self.previous(step)
        summary = {'schema': SCHEMA, 'step': step, 'task_successes': score['task_successes'],
                   'cases': len(self.rows), 'task_success_rate': score['task_success_rate'],
                   'outcome_counts': score['outcome_counts'], 'complete': score['complete'],
                   'eligible': eligible,
                   'collector_exit_codes': codes, 'prompt_rendering': self.prompt_rendering,
                   'packet_sha256': self.spec['packet']['sha256'], 'case_indices': self.spec['cases'],
                   'reference_successes': self.spec.get('reference_successes'),
                   'min_successes': self.spec['min_successes'],
                   'repeat_successes': repeats or None, 'repeat_noise': repeat_noise,
                   'noise_successes': noise,
                   'previous_step': previous.get('step') if previous else None,
                   'previous_successes': previous.get('task_successes') if previous else None,
                   'regressed': bool(previous) and score['task_successes'] < previous['task_successes'] - noise,
                   'code_commit': _git_head(),
                   'seconds': time.time() - started, 'run': str(run)}
        (run / 'task-score.json').write_text(json.dumps({**summary, 'cases_detail': score['cases']}, indent=1) + '\n')
        with (self.out / 'task-eval.jsonl').open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        print(json.dumps({'event': 'task_eval', **{k: summary[k] for k in
                          ('step', 'task_successes', 'cases', 'outcome_counts', 'eligible', 'previous_successes',
                           'repeat_noise', 'noise_successes', 'regressed', 'seconds')}}), flush=True)
        return summary


def task_gate(summary: dict | None, spec: dict | None) -> dict | None:
    """The qualification criterion on the final in-run task evaluation (None when no task eval is declared):
    eligible, at least the base (``reference_successes``) and ``min_successes``, and not below the previous eligible
    evaluation minus ``noise_successes``."""
    if spec is None:
        return None
    if summary is None:
        return {'passed': False, 'reason': 'no final task evaluation', 'min_successes': spec['min_successes']}
    successes, reference = summary['task_successes'], spec.get('reference_successes')
    reasons = []
    if not summary.get('eligible'):
        reasons.append('final task evaluation incomplete or with infrastructure errors')
    if reference is None:
        reasons.append('base reference_successes unmeasured for this line')
    elif successes < reference:
        reasons.append('below the base model')
    if successes < spec['min_successes']:
        reasons.append('below min_successes')
    if summary.get('regressed'):
        reasons.append('below the previous evaluation minus noise')
    return {'passed': not reasons, 'reasons': reasons, 'task_successes': successes, 'cases': summary['cases'],
            'reference_successes': reference, 'min_successes': spec['min_successes'],
            'noise_successes': summary.get('noise_successes', spec['noise_successes']),
            'previous_successes': summary.get('previous_successes'),
            'previous_step': summary.get('previous_step'), 'eligible': summary.get('eligible'),
            'step': summary.get('step')}


def main(argv=None) -> int:
    """Standalone: evaluate a base model (or a direct heads checkpoint) with the in-run evaluator, e.g. to measure a
    line's ``reference_successes`` or smoke the path: ``python -m natlang_neuralese.eval.task_cadence --base DIR
    --spec JSON --out DIR``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--base', default=None)
    parser.add_argument('--heads', default=None)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    from ..serve import load_engine

    torch.set_num_threads(args.threads)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args.dtype]
    engine = load_engine(args.base, None, args.heads, None, None, args.device, None, dtype=dtype,
                         prompt_rendering=args.prompt_rendering)
    spec = parse_task_eval(args.spec)
    summary = TaskEval(spec, args.out, engine, prompt_rendering=args.prompt_rendering).run(args.step)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
