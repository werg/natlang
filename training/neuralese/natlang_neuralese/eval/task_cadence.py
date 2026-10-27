"""In-run native task-success evaluation (owner decision 2026-10-12, plans/neuralese/DECISIONS.md "Qualification is task success
plus drift").

A fixed subset of the protected held task packet is executed for real at the training eval cadence, against the
native checkpoint being trained, in the trainer's own process: its native serving engine is served on a loopback
port for the duration of the evaluation, and the shared TS collector
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
  prompt_rendering  serving prompt policy for the generative evaluation; default: full, whatever the stage trains on
                  (owner 2026-10-13: models trained without the system prompt for economy are still evaluated and served
                  with the proper full system prompt; program-header only as an explicit setting)
  case_seconds    wall budget per task (collection resource limit, not a model label; default 7200, set from the
                  2026-10-13 LFM2.5-2.6B measurement in plans/neuralese/DECISIONS.md "Task-eval budgets")
  max_model_requests / max_turns   per-task request and turn budgets (default 128 / 160)
  workers         tasks executed concurrently (default 8; the engine batches their requests)
  min_successes   absolute floor on the final evaluation's task successes (default 0)
  reference_successes  the line's base model's successes on this subset under its native rendering
  noise_successes  declared run-to-run noise of the success count on this subset (from repeat runs)
  noise_repeats   runs of the stage's first evaluation (default 2): their spread is the measured noise; the bar
                  uses the larger of the declared and the measured noise
  temperature     sampling temperature the collector sends (default 0: the server's greedy argmax). At 0 a repeat
                  run replays the same actions, so its noise is zero by construction; above 0 each repeat run gets
                  its own seeds (REPETITION_DRAGNET_20261013.md)

The pass bar is relative (coordinator 2026-10-12): the final evaluation is eligible, reaches the base
(``reference_successes``; an unmeasured reference fails the gate) and ``min_successes``, and is not below the
previous eligible evaluation of the stage minus ``noise_successes``. Every evaluation records the same comparison
(``regressed``) for monitoring; only the final one gates.
  runtime         TS runtime directory with dist/teacher (default: the live checkout's ts-host, the newest code)
  native asynchronous evaluation uses the checkpoint-addressed export service in ``eval.native_async``; synchronous
  cadence evaluation remains available in the trainer.
  max_consecutive_failures  ineligible evaluations in a row (infrastructure errors, unresolved tasks, a collector or
                  serving exception) before the training run stops (default 3). A single failed evaluation is recorded
                  in task-eval.jsonl and the next cadence evaluates again: a transient eval failure must not end a
                  multi-day run, a persistent one must not pass silently (DGX/Pop 2026-10-12).
  runtime_failure_policy  stop (legacy default), diagnostic-partial, or report. Report preserves every
                  incomplete result/counter as unscored monitoring and continues routine training;
                  mandatory current-weight final/foundation checks remain strict.
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

from .task_packet import accounted

DEFAULT_CASES = tuple(range(0, 48, 3))
SCHEMA = 'natlang.in-run-task-eval/1'


def wait_collectors(children, queues, launch, *, min_free_mib, max_workers=None, stop_requested=None):
    """Run exact case queues, optionally refilling a bounded set of collector slots.

    With ``max_workers`` the queue list is a stable case order and children starts
    empty. A storage-paused case stays pending while its slot runs other cases;
    only that exact queue can be relaunched after the disk floor is restored.
    Without it, callers may pass the historical one-child-per-queue shape.
    """
    snapshots = [queue.read_bytes() for queue, _, _ in queues]
    if max_workers is None:
        if len(children) != len(queues):
            raise ValueError('task collector child count must match fixed queues')
        max_workers = len(queues)
        running = {index: children[index] for index in range(len(children))}
        launched = set(running)
    else:
        if type(max_workers) is not int or max_workers < 1 or children:
            raise ValueError('rolling task dispatch needs an empty child list and positive worker count')
        running, launched = {}, set()
    codes = [None] * len(queues)
    pending = set(range(len(queues)))
    paused = set()
    interrupted = False
    journal_error = None

    def check_pause(index):
        queue, journal, entries = queues[index]
        events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        last = events[-1] if events else {}
        if last.get('event') != 'storage_pause':
            return False  # Operator drains and all other exit-75 reasons are terminal.
        finished = {row['key'] for row in events if row.get('event') == 'finish'}
        remaining = [row['key'] for row in entries if row['key'] not in finished]
        if (queue.read_bytes() != snapshots[index]
                or last.get('minimum_free_mib') != min_free_mib
                or not remaining or last.get('key') != remaining[0]
                or any(row.get('event') == 'start' and row.get('key') == last['key'] for row in events)):
            raise ValueError('task evaluation storage pause does not match its exact unstarted queue')
        return True

    while pending:
        if not interrupted and stop_requested is not None and stop_requested():
            interrupted = True
            for index in sorted(pending):
                queue, journal, entries = queues[index]
                event = {'event': 'operational_stop', 'time': time.time(),
                         'queue': str(queue), 'started': index in running,
                         'keys': [entry['key'] for entry in entries]}
                try:
                    with journal.open('a') as stream:
                        stream.write(json.dumps(event) + '\n'); stream.flush(); os.fsync(stream.fileno())
                except OSError as error:
                    if journal_error is None: journal_error = error
                if index in running:
                    child = running[index][0]
                    if child.poll() is None:
                        try: child.terminate()
                        except ProcessLookupError: pass
                else:
                    codes[index] = 75
            pending.intersection_update(running)
            paused.clear()
        for index in tuple(sorted(running)):
            child, log = running[index]
            code = child.poll()
            if code is None:
                continue
            log.close()
            del running[index]
            if not interrupted and code == 75 and min_free_mib and check_pause(index):
                paused.add(index)
                continue
            codes[index] = code
            pending.remove(index)

        free_slots = max_workers - len(running)
        if free_slots and pending and not interrupted:
            disk_ok = (not min_free_mib or
                       shutil.disk_usage(queues[0][1].parent).free // (1024 * 1024) >= min_free_mib)
            if disk_ok:
                # A disk-paused case waits behind untouched work once the floor
                # returns; it never holds a slot while other cases remain.
                candidates = ([index for index in sorted(pending)
                               if index not in running and index not in launched] +
                              [index for index in sorted(paused) if index in pending and index not in running])
                for index in candidates[:free_slots]:
                    if stop_requested is not None and stop_requested(): break
                    queue, journal, entries = queues[index]
                    if index in paused and not check_pause(index):
                        # An operator may drain a paused case while other work
                        # runs. Recheck its durable journal before resuming it.
                        codes[index] = 75
                        paused.discard(index)
                        pending.remove(index)
                        continue
                    if queue.read_bytes() != snapshots[index]:
                        raise ValueError('task evaluation queue changed before collector launch')
                    running[index] = launch(queue, journal, entries)
                    launched.add(index)
                    paused.discard(index)
        if pending:
            time.sleep(0.2 if running or stop_requested is not None else 30)
    if journal_error is not None: raise journal_error
    return codes


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
    for key, default in (('case_seconds', 7200), ('max_model_requests', 128), ('max_turns', 160), ('workers', 8),
                         ('min_successes', 0), ('noise_successes', 0), ('noise_repeats', 2),
                         ('max_consecutive_failures', 3), ('identical_turn_limit', 0),
                         ('min_free_mib', 0)):
        spec.setdefault(key, default)
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < (
                0 if key in ('min_successes', 'noise_successes', 'identical_turn_limit', 'min_free_mib') else 1):
            raise ValueError(f'task_eval.{key} must be a positive integer')
    reference = spec.setdefault('reference_successes', None)
    if reference is not None and (isinstance(reference, bool) or not isinstance(reference, int) or reference < 0):
        raise ValueError('task_eval.reference_successes must be a nonnegative integer or null (unmeasured)')
    if not isinstance(spec.setdefault('neuralese_server', False), bool):
        raise ValueError('task_eval.neuralese_server must be true or false')
    if spec.get('prompt_rendering') not in (None, 'full', 'program-header'):
        raise ValueError('task_eval.prompt_rendering must be full or program-header')
    spec.setdefault('prompt_rendering', 'full')
    if spec.setdefault('runtime_failure_policy', 'stop') not in ('stop', 'diagnostic-partial', 'report'):
        raise ValueError('task_eval.runtime_failure_policy must be stop, diagnostic-partial or report')
    temperature = spec.setdefault('temperature', 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError('task_eval.temperature must be a number in [0, 2]')
    return spec


def _task_eval_runtime_files(repo,runtime):
    """Return per-file custody for the full executing TS and Python closures."""
    from ..common.hashing import sha256_file_hex_cached
    from ..common.runtime_inventory import task_runtime_inventory
    from ..train.native_resident import code_inventory
    repo=Path(repo).resolve(strict=True);runtime=Path(runtime).resolve(strict=True)
    files={}
    for relative,digest in task_runtime_inventory(runtime).items():
        path=(runtime/relative).resolve(strict=True)
        files[str(path)]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
    collector=(repo/'scripts/run_bonsai_queue.py').resolve(strict=True)
    files[str(collector)]={'path':str(collector),'bytes':collector.stat().st_size,
        'sha256':sha256_file_hex_cached(collector)}
    package=(repo/'training/neuralese/natlang_neuralese').resolve(strict=True)
    for relative,digest in code_inventory(package).items():
        path=(package/relative).resolve(strict=True)
        files[str(path)]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
    return [files[key] for key in sorted(files)]


def task_eval_runtime_identity(value):
    """Bind complete task runtime inventories and every executing Python module."""
    from ..common.paths import root as logical_root
    repo=logical_root('repo').resolve(strict=True)
    runtime=Path(value.get('runtime') or repo/'ts-host').resolve(strict=True)
    return {'schema':'natlang.task-eval-runtime-identity/1','repo':str(repo),
        'runtime':str(runtime),'code_commit':_git_head(),
        'files':_task_eval_runtime_files(repo,runtime)}


def task_eval_policy_identity(value):
    """Hash the exact task execution contract, excluding only its comparison baseline.

    A baseline is meaningful only for the same packet, selected cases, prompt
    policy, collector budgets and runtime options. The reference count and its
    provenance are outputs of baseline measurement, so they are excluded.
    """
    from ..common.hashing import canonical_json_sha256_hex
    spec = parse_task_eval(value)
    if spec is None:
        raise ValueError('task evaluation policy identity requires a task_eval declaration')
    normalized = dict(spec)
    normalized.pop('reference_successes', None)
    normalized.pop('reference_provenance', None)
    normalized['runtime_identity']=task_eval_runtime_identity(spec)
    return canonical_json_sha256_hex(normalized)


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
        from ..common.paths import root as logical_root
        return subprocess.check_output(['git', '-C', str(logical_root('repo')), 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_native_transport(spec):
    """Native values require the collector's state store and native client."""
    if spec.get('neuralese_server') is not True:
        raise ValueError('native task evaluation requires neuralese_server=true for state transport')
    return spec


class TaskEval:
    """Serve ``engine`` on loopback and execute the fixed held subset against it."""

    def __init__(self, spec: dict, out: Path, engine, *, prompt_rendering: str, external_actor=None, stop_requested=None):
        from ..common.hashing import sha256_file_hex
        from .task_packet import read_jsonl

        if getattr(engine, 'native_state_enabled', False):
            require_native_transport(spec)
        self.spec, self.out, self.engine = spec, Path(out), engine
        self.external_actor = external_actor
        self.stop_requested = stop_requested
        self.prompt_rendering = spec.get('prompt_rendering') or prompt_rendering or 'full'
        from ..common.paths import root as logical_root
        # Repo-relative paths resolve against the live checkout (a training run executes a code snapshot that
        # carries neither runs/ nor the built ts-host).
        self.repo = logical_root('repo')
        collector = self.repo / 'scripts' / 'run_bonsai_queue.py'
        runtime = Path(spec.get('runtime') or self.repo / 'ts-host')
        if not collector.is_file():
            raise FileNotFoundError(f'task_eval collector missing from live checkout: {collector}')
        if not (runtime / 'dist' / 'teacher' / 'cli.js').is_file():
            raise FileNotFoundError(f'task_eval TS runtime missing built collector: {runtime}')
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

    def operational_stop_requested(self):
        callback = getattr(self, 'stop_requested', None)
        return callback is not None and bool(callback())

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

    def _queues(self, run: Path, repeat: int = 0) -> list[tuple[Path, Path, list[dict]]]:
        workers = min(self.spec['workers'], len(self.rows))
        queues = []
        for index in range(len(self.rows)):
            worker = index % workers
            directory = run / f'worker-{worker:02d}' / f'case-{index:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            case = directory
            entries = [{'index': index, 'count': 1, 'key': f'task-{index:04d}',
                        'source': str(self.subset), 'jobs': str(case / 'jobs'),
                        'log': str(case / 'collector.log'), 'output': str(case / 'results.jsonl'),
                        'case_seconds': self.spec['case_seconds'], 'context_tokens': 32768,
                        'max_model_requests': self.spec['max_model_requests'],
                        'prompt_rendering': self.prompt_rendering,
                        'max_turns': self.spec['max_turns'], 'request_retries': 1,
                        'transport_retries': 0,
                        'seed': 400000 + self.spec['cases'][index] + (100000 * repeat if self.spec.get('temperature') else 0),
                        'text_neuralese_emulation': False,
                        'temperature': self.spec['temperature'],
                        'neuralese_server': self.spec.get('neuralese_server',False)}]
            queue = directory / 'queue.jsonl'
            queue.write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries), encoding='utf-8')
            queues.append((queue, directory / 'journal.jsonl', entries))
        return queues

    def _execute(self, run: Path, repeat: int = 0) -> tuple[dict, list[int]]:
        """Serve the engine and run every subset task once into ``run``; the exact score and collector exit codes."""
        from ..serve.http import serve
        from .task_packet import read_jsonl, score_rows

        external = getattr(self, 'external_actor', None)
        if run.exists() and external is None:
            shutil.rmtree(run)  # a rerun of the same step (resume) replaces an interrupted attempt
        run.mkdir(parents=True, exist_ok=external is not None)
        engine, hf = self.engine, getattr(getattr(self.engine, 'backbone', None), 'hf', None)
        was_training = bool(hf is not None and hf.training)
        previous_mode = engine.prompt_rendering
        server = None
        try:
            if external is None:
                if hf is not None:
                    hf.eval()
                engine.prompt_rendering = self.prompt_rendering
                # Engine.stop() leaves its stop flag set; reset it once per evaluation.
                import queue as queue_module
                engine._stopping, engine._incoming, engine._active = False, queue_module.Queue(), []
                engine.start()
                server = serve(engine, '127.0.0.1', 0)
                threading.Thread(target=server.serve_forever, name='task-eval-http', daemon=True).start()
                url = f'http://127.0.0.1:{server.server_address[1]}'
            else:
                url = external['url']
            env = node_path_env()
            runtime = Path(self.spec.get('runtime') or self.repo / 'ts-host')
            from ..common.runtime_inventory import task_source_reviews
            env['NATLANG_SOURCE_REVIEWS'] = str(task_source_reviews(runtime))
            children = []
            queues = self._queues(run, repeat)
            coefficient_runtime=getattr(engine,'native_coefficient_policy_runtime',None)
            coefficient_args=[]
            if coefficient_runtime is not None:
                config=run/'native-coefficient-request.json'
                from ..common.jsonio import write_json_durable
                write_json_durable(config,{'x_natlang_native_coefficient_policy':coefficient_runtime})
                coefficient_args=['--chat-request-config',str(config)]
            def launch(queue, journal, entries):
                log = (queue.parent / 'supervisor.log').open('a')
                # The collector's default model ID is the local Bonsai teacher. This endpoint serves the
                # student engine; bind its identity in requests and trajectory provenance instead.
                return (subprocess.Popen(
                    [sys.executable, str(self.repo / 'scripts' / 'run_bonsai_queue.py'), str(queue), str(journal),
                     '--runtime', str(self.spec.get('runtime') or self.repo / 'ts-host'), '--server', url,
                     '--model-id', engine.model_name,
                     '--chat-completions-url', url + '/v1/chat/completions', '--model-concurrency', '2',
                     '--min-free-mib', str(self.spec.get('min_free_mib', 0)),
                     '--identical-turn-limit', str(self.spec['identical_turn_limit']), *coefficient_args,
                     *(['--neuralese-server'] if self.spec.get('neuralese_server') else [])],
                    cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT), log)
            codes = wait_collectors([], queues, launch,
                                    min_free_mib=self.spec.get('min_free_mib', 0),
                                    max_workers=min(self.spec['workers'], len(queues)),
                                    stop_requested=self.operational_stop_requested)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if external is None:
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
                            finishes[f'{queue.parent.parent.name}/{event["key"]}'] = event
            for entry in entries:
                keys[entry['index']] = f'{queue.parent.parent.name}/{entry["key"]}'
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
        operational_stop = self.operational_stop_requested()
        eligible = accounted(score) and not operational_stop
        # Noise from repeat runs (coordinator 2026-10-12): the first evaluation of a stage runs the same model again
        # (noise_repeats total runs); the spread of the success counts is this stage's measured noise.
        repeats, repeat_noise = [], None
        if eligible and self.measured_noise() is None and self.spec.get('noise_repeats', 2) > 1:
            for index in range(1, self.spec.get('noise_repeats', 2)):
                if self.operational_stop_requested(): break
                again, again_codes = self._execute(run.with_name(run.name + f'-repeat{index}'), index)
                if accounted(again):
                    repeats.append(again['task_successes'])
            if repeats:
                counts = [score['task_successes'], *repeats]
                repeat_noise = max(counts) - min(counts)
        operational_stop = operational_stop or self.operational_stop_requested()
        eligible = eligible and not operational_stop
        measured = repeat_noise if repeat_noise is not None else self.measured_noise()
        noise = max(self.spec['noise_successes'], measured or 0)
        previous = self.previous(step)
        coefficient_runtime=getattr(getattr(self,'engine',None),'native_coefficient_policy_runtime',None)
        summary = {'schema': SCHEMA, 'step': step, 'task_successes': score['task_successes'],
                   **({'coefficient_policy_sha256':coefficient_runtime['policy_sha256']} if coefficient_runtime is not None else {}),
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
                   'regressed': eligible and bool(previous) and score['task_successes'] < previous['task_successes'] - noise,
                   'code_commit': _git_head(),
                   'seconds': time.time() - started, 'run': str(run)}
        if operational_stop:
            summary.update(operational_stop=True, qualification_deferred=True)
        if not eligible:
            summary.update(observed_task_successes=score['task_successes'],
                           observed_regressed=bool(previous) and score['task_successes'] < previous['task_successes'] - noise,
                           task_successes=None, task_success_rate=None, model_outcome='unscored')
        if getattr(self,'evaluation_provenance',None):
            from .native_async import verify_environment
            summary['evaluation_provenance']=verify_environment(self.evaluation_provenance)
            summary['historical']=True
            summary['qualifies_current_weights']=False
        (run / 'task-score.json').write_text(json.dumps({**summary, 'cases_detail': score['cases']}, indent=1) + '\n')
        with (self.out / 'task-eval.jsonl').open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        print(json.dumps({'event': 'task_eval', **{k: summary[k] for k in
                          ('step', 'task_successes', 'cases', 'outcome_counts', 'eligible', 'previous_successes',
                           'repeat_noise', 'noise_successes', 'regressed', 'seconds')}}), flush=True)
        return summary

    def consecutive_failures(self) -> int:
        """Ineligible evaluations at the end of this stage's log (a resumed run continues the count)."""
        log = self.out / 'task-eval.jsonl'
        count = 0
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # A new held packet or case subset starts a new comparison. Historical
                # infrastructure failures on a superseded packet cannot exhaust this
                # packet's retry allowance.
                if (row.get('packet_sha256') != self.spec['packet']['sha256'] or
                        row.get('case_indices') != self.spec['cases']):
                    count = 0
                else:
                    if row.get('superseded') or row.get('skipped') or row.get('operational_stop'):
                        continue  # a dropped request is no evaluation: neither a failure nor a success
                    count = 0 if row.get('eligible') else count + 1
        return count

    def final_at(self, step: int) -> dict:
        """Recover a final evaluation after an exact-step resume without new updates.

        A completed optimizer checkpoint can precede a failed collector. Reuse
        an eligible result for the same packet and step, or rerun the collector
        until it yields an eligible final result (subject to its failure cap).
        """
        log = self.out / 'task-eval.jsonl'
        if log.is_file():
            for line in reversed(log.read_text().splitlines()):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('step') == step and row.get('operational_stop') is True:
                    break  # Do not reuse an earlier publication of this interrupted attempt.
                if (row.get('step') == step and row.get('eligible') is True and
                        row.get('packet_sha256') == self.spec['packet']['sha256'] and
                        row.get('case_indices') == self.spec['cases']):
                    return row
        summary = self.cadence(step, final=True)
        while not summary['eligible'] and not self.operational_stop_requested():
            summary = self.cadence(step, final=True)
        return summary

    def cadence(self, step: int, *, final=False) -> dict:
        """One in-run evaluation for the training loop: the summary of ``run``. An ineligible evaluation or an
        exception from the collector or serving stack is recorded and returned as an ineligible summary, so the next
        cadence evaluates again. The explicit report policy keeps routine failures as unscored monitoring;
        mandatory final checks and other policies retain the consecutive-failure bound."""
        started = time.time()
        try:
            summary = self.run(step)
        except Exception as error:  # noqa: BLE001 - any eval infrastructure failure is recorded, then counted
            summary = {'schema': SCHEMA, 'step': step, 'task_successes': None, 'task_success_rate': None, 'model_outcome': 'unscored', 'cases': len(self.rows),
                       'outcome_counts': None, 'eligible': False, 'error': f'{type(error).__name__}: {error}',
                       'packet_sha256': self.spec['packet']['sha256'], 'case_indices': self.spec['cases'],
                       **({'coefficient_policy_sha256':getattr(self,'engine').native_coefficient_policy_runtime['policy_sha256']}
                          if getattr(getattr(self,'engine',None),'native_coefficient_policy_runtime',None) is not None else {}),
                       'code_commit': _git_head(), 'seconds': time.time() - started,
                       'run': str(self.root / f'step-{step:07d}')}
            if self.operational_stop_requested():
                summary.update(operational_stop=True, qualification_deferred=True)
            with (self.out / 'task-eval.jsonl').open('a') as handle:
                handle.write(json.dumps(summary) + '\n')
        if self.operational_stop_requested():
            already_recorded = summary.get('operational_stop') is True
            summary.update(operational_stop=True, qualification_deferred=True, eligible=False, regressed=False,
                           observed_task_successes=summary.get('observed_task_successes',summary.get('task_successes')),
                           task_successes=None, task_success_rate=None, model_outcome='unscored')
            if not already_recorded:
                # The stop can arrive after run() persisted its score. Append an
                # explicit correction; never return a passed receipt while stopped.
                with (self.out / 'task-eval.jsonl').open('a') as handle:
                    handle.write(json.dumps(summary) + '\n')
            return summary
        if summary['eligible']:
            return summary
        failures = self.consecutive_failures()
        print(json.dumps({'event': 'task_eval_failed', 'step': step, 'consecutive_failures': failures,
                          'max_consecutive_failures': self.spec['max_consecutive_failures'],
                          'outcome_counts': summary.get('outcome_counts'), 'error': summary.get('error'),
                          'run': summary.get('run')}), flush=True)
        if not final and (self.spec.get('runtime_failure_policy') == 'report' or partial_runtime_diagnostic(summary, self.spec)):
            print(json.dumps({'event': 'task_eval_runtime_degraded', 'step': step,
                              'consecutive_failures': failures, 'eligible': False,
                              'policy': self.spec.get('runtime_failure_policy', 'stop'), 'qualification': 'unchanged; incomplete evaluation'}),
                  flush=True)
            return summary
        if failures >= self.spec['max_consecutive_failures']:
            raise RuntimeError(f'task_eval failed {failures} times in a row (last at step {step}: '
                               f'{summary.get("error") or summary.get("outcome_counts")}); see {summary.get("run")}')
        return summary

    def landed(self, step: int) -> dict | None:
        """The row recorded for ``step`` on this packet and subset (eligible or not), or None."""
        log = self.out / 'task-eval.jsonl'
        found = None
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get('step') == step and row.get('packet_sha256') == self.spec['packet']['sha256'] and
                        row.get('case_indices') == self.spec['cases']):
                    found = row
        return found

    def record_failure(self, step: int, error: str) -> dict:
        """Record an ineligible asynchronous evaluation result for this exact step."""
        summary = {'schema': SCHEMA, 'step': step, 'task_successes': None, 'task_success_rate': None, 'model_outcome': 'unscored', 'cases': len(self.rows),
                   'outcome_counts': None, 'eligible': False, 'error': error, 'async': True,
                   'packet_sha256': self.spec['packet']['sha256'], 'case_indices': self.spec['cases'],
                   'code_commit': _git_head(), 'seconds': 0.0, 'run': str(self.root / f'step-{step:07d}')}
        if getattr(self,'evaluation_provenance',None):
            summary.update(task_successes=None,task_success_rate=None,complete=False,
                historical=True,qualifies_current_weights=False,model_outcome='unscored',
                evaluation_provenance=self.evaluation_provenance)
        with (self.out / 'task-eval.jsonl').open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        return summary


ASYNC_SCHEMA = 'natlang.in-run-task-eval-request/1'


def request_queue() -> Path:
    """Pointers to pending native async task evaluations, so the dispatcher never walks the runs tree."""
    from ..common.paths import root as logical_root
    return logical_root('data_nvme') / 'task-eval-queue'


def request_weights(request_dir: Path) -> Path:
    try:
        recorded = json.loads((Path(request_dir) / 'request.json').read_text()).get('weights')
    except (OSError, ValueError):
        recorded = None
    return Path(recorded) if recorded else Path(request_dir) / 'weights.pt'


def drop_request(request_dir: Path) -> None:
    """Delete a request: its snapshot (wherever it lives) and its dir; the queue pointer goes with the dir."""
    from ..train.checkpoint_offload import PART_SUFFIX
    request_dir = Path(request_dir)
    weights = request_weights(request_dir)
    if weights.parent != request_dir:
        weights.unlink(missing_ok=True)
        weights.with_name(weights.name + PART_SUFFIX).unlink(missing_ok=True)
        try:
            weights.parent.rmdir()
        except OSError:
            pass
    shutil.rmtree(request_dir, ignore_errors=True)


def restore_engine(request: dict, *, dtype=None):
    """Restore the exact native checkpoint bound to an asynchronous evaluation request."""
    from ..common.attention_backend import configure_sdpa_backend
    from ..serve import load_native_engine
    if request.get('native_state') is not True:
        raise ValueError('task evaluation requests must bind a native-state checkpoint')
    configure_sdpa_backend(request.get('sdpa_backend', 'default'))
    from .native_async import file_binding, verify_request
    provenance=request['evaluation_provenance']
    verify_request(request)
    weights=request_weights(Path(request['dir']))
    actual=file_binding(weights)
    expected=provenance['checkpoint']
    if any(actual[key]!=expected[key] for key in ('sha256','bytes')):
        raise ValueError('native async eval checkpoint bytes changed')
    engine=load_native_engine(str(weights),base=request['base'],device=request['device'],
        prompt_rendering=request['prompt_rendering'])
    saved=provenance['model_config']
    for key in ('architecture','rank','marker_ids','tokenizer_fingerprint','base_inventory_sha256',
                'dtype','sparse_row_dtype','token_override_ids','token_override_dtype','native_linear_precision','native_attention_precision','precision'):
        default='ieee' if key in ('native_linear_precision','native_attention_precision') else None
        expected_value=saved.get(key,default)
        if engine.model_config.get(key,default)!=expected_value:
            raise ValueError('native async eval model contract changed: '+key)
    if engine.native_identity['checkpoint_sha256']!=expected['sha256']:
        raise ValueError('native async eval restored another checkpoint identity')
    return engine


def run_request(request_dir: Path, *, engine_factory=restore_engine) -> dict:
    """The asynchronous job: evaluate one snapshot and append its row (eligible or not) to the stage's log, then
    delete the snapshot. A rerun after the row landed (a requeued unit) only cleans up."""
    request_dir = Path(request_dir)
    request = {**json.loads((request_dir / 'request.json').read_text()), 'dir': str(request_dir)}
    step = request['step']
    evaluator = TaskEval(request['spec'], Path(request['stage_out']), None,
                         prompt_rendering=request['prompt_rendering'])
    evaluator.evaluation_provenance=request.get('evaluation_provenance')
    summary = evaluator.landed(step)
    if summary is not None and request.get('native_state'):
        if (summary.get('evaluation_provenance') or {}).get('source_checkpoint_sha256') != request['evaluation_provenance']['source_checkpoint_sha256']:
            raise ValueError('native async landed result belongs to different weights')
    if summary is None:
        started = time.time()
        try:
            evaluator.evaluation_provenance=request.get('evaluation_provenance')
            evaluator.engine = engine_factory(request)
            if request.get('native_state'):
                evaluator.evaluation_provenance={**evaluator.evaluation_provenance,
                    'native_identity':evaluator.engine.native_identity,
                    'torch_version':__import__('torch').__version__,
                    'cuda_device':(__import__('torch').cuda.get_device_name()
                                   if str(request['device']).startswith('cuda') else None),
                    'request_sha256':__import__('hashlib').sha256((request_dir/'request.json').read_bytes()).hexdigest()}
            summary = evaluator.run(step)
        except Exception as error:  # noqa: BLE001 - recorded as an ineligible evaluation, counted by the trainer
            summary = evaluator.record_failure(step, f'{type(error).__name__}: {error}')
            summary['seconds'] = time.time() - started
        finally:
            evaluator.engine = None
    drop_request(request_dir)  # storage rules: the snapshot goes once its row landed
    return summary


def task_quality_reasons(summary, spec):
    """Existing capability comparisons, separate from runtime score eligibility."""
    successes, reference = summary.get('task_successes'), spec.get('reference_successes')
    reasons = []
    if successes is None:
        return ['task evaluation unscored']
    if reference is None:
        reasons.append('base reference_successes unmeasured for this line')
    elif successes < reference:
        reasons.append('below the base model')
    if successes < spec['min_successes']:
        reasons.append('below min_successes')
    if summary.get('regressed'):
        reasons.append('below the previous evaluation minus noise')
    return reasons


def partial_runtime_diagnostic(summary, spec):
    """Explicit diagnostic continuation; never changes the score or qualification."""
    if (spec.get('runtime_failure_policy') != 'diagnostic-partial' or summary.get('error')
            or summary.get('eligible')):
        return False
    counts, expected = summary.get('outcome_counts'), summary.get('cases')
    if (not isinstance(counts, dict) or type(expected) is not int or expected < 1
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != expected):
        return False
    unscored = counts.get('infrastructure_error', 0) + counts.get('unresolved', 0)
    known = expected - unscored
    if not unscored or known <= expected / 2:
        return False
    known_counts = {**counts, 'infrastructure_error': 0, 'unresolved': 0}
    if not accounted({'cases_expected': known, 'outcome_counts': known_counts}):
        return False
    diagnostic = summary
    if summary.get('task_successes') is None:
        diagnostic = {**summary, 'task_successes': summary.get('observed_task_successes'),
                      'regressed': summary.get('observed_regressed', summary.get('regressed', False))}
    return not task_quality_reasons(diagnostic, spec)


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
    reasons.extend(task_quality_reasons(summary, spec))
    return {'passed': not reasons, 'reasons': reasons, 'task_successes': successes, 'cases': summary['cases'],
            'reference_successes': reference, 'min_successes': spec['min_successes'],
            'noise_successes': summary.get('noise_successes', spec['noise_successes']),
            'previous_successes': summary.get('previous_successes'),
            'previous_step': summary.get('previous_step'), 'eligible': summary.get('eligible'),
            'step': summary.get('step')}


def main(argv=None) -> int:
    """Standalone: evaluate a native-state checkpoint with the in-run evaluator, e.g. to measure a
    line's ``reference_successes`` or smoke the path: ``python -m natlang_neuralese.eval.task_cadence --base DIR
    --spec JSON --out DIR``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--base', default=None)
    parser.add_argument('--native-checkpoint', default=None)
    parser.add_argument('--async-request', type=Path, default=None,
                        help='run one asynchronous in-run evaluation request (task-eval/async/step-N) and exit')
    parser.add_argument('--spec')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    from ..serve import load_native_engine

    if args.async_request is not None:
        torch.set_num_threads(args.threads)
        summary = run_request(args.async_request)
        print(json.dumps({'event': 'task_eval_async_done', **{k: summary.get(k) for k in
                          ('step', 'task_successes', 'cases', 'eligible', 'error', 'seconds')}}), flush=True)
        return 0
    if not args.spec or args.out is None or not args.native_checkpoint:
        parser.error('--spec, --out, and --native-checkpoint are required without --async-request')

    torch.set_num_threads(args.threads)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args.dtype]
    engine = load_native_engine(args.native_checkpoint, base=args.base, device=args.device, dtype=dtype,
                                prompt_rendering=args.prompt_rendering)
    spec = parse_task_eval(args.spec)
    summary = TaskEval(spec, args.out, engine, prompt_rendering=args.prompt_rendering).run(args.step)
    print(json.dumps(summary, indent=1))
    return 0 if summary['eligible'] else 1


if __name__ == '__main__':
    sys.exit(main())
