"""In-run task-success evaluation (owner decision 2026-10-12): spec parsing, serving policy and the qualification gate."""
import pytest

from natlang_neuralese.eval.task_cadence import DEFAULT_CASES, parse_task_eval, task_gate, training_prompt_rendering


def test_spec_defaults_to_a_fixed_sixteen_task_subset_and_validates():
    spec = parse_task_eval('{"packet": {"path": "p.jsonl", "sha256": "' + 'a' * 64 + '"}}')
    assert spec['cases'] == list(DEFAULT_CASES) and len(spec['cases']) == 16
    assert spec['min_successes'] == 0 and spec['noise_successes'] == 0 and spec['workers'] == 8
    assert (spec['case_seconds'], spec['max_model_requests'], spec['max_turns']) == (7200, 128, 160)
    assert spec['reference_successes'] is None
    assert parse_task_eval(None) is None
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p'}})
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p', 'sha256': 'x'}, 'cases': [1, 1]})


def test_serving_policy_follows_the_rows_the_stage_trains_on():
    assert training_prompt_rendering([{'prompt_rendering': 'program-header'}] * 3 + [{}]) == 'program-header'
    assert training_prompt_rendering([{}, {}]) == 'full'


def test_task_gate_is_relative_to_the_base_and_the_previous_evaluation():
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'reference_successes': 3, 'noise_successes': 2})
    final = lambda n, **k: {'eligible': True, 'task_successes': n, 'cases': 16, 'step': 9, **k}
    assert task_gate(None, None) is None
    assert task_gate(None, spec)['passed'] is False
    assert task_gate(final(3), spec)['passed'] is True
    assert task_gate(final(2), spec)['reasons'] == ['below the base model']
    assert task_gate(final(5, previous_successes=8, regressed=True), spec)['passed'] is False
    assert task_gate(final(6, previous_successes=8, regressed=False), spec)['passed'] is True
    assert task_gate({**final(9), 'eligible': False}, spec)['passed'] is False
    unmeasured = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}})
    assert task_gate(final(9), unmeasured)['reasons'] == ['base reference_successes unmeasured for this line']


def test_previous_evaluation_and_regression_are_read_from_the_stage_log(tmp_path):
    import json
    from natlang_neuralese.eval.task_cadence import TaskEval
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_successes': 1})
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out = spec, tmp_path
    rows = [{'step': 100, 'eligible': True, 'task_successes': 5, 'packet_sha256': 's', 'case_indices': [0]},
            {'step': 200, 'eligible': False, 'task_successes': 9, 'packet_sha256': 's', 'case_indices': [0]},
            {'step': 150, 'eligible': True, 'task_successes': 7, 'packet_sha256': 'other', 'case_indices': [0]}]
    (tmp_path / 'task-eval.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert evaluator.previous(300)['step'] == 100
    assert evaluator.previous(100) is None


def test_the_first_evaluation_repeats_to_measure_noise_and_later_ones_reuse_it(tmp_path):
    from natlang_neuralese.eval.task_cadence import TaskEval
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_repeats': 3})
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out, evaluator.root, evaluator.rows = spec, tmp_path, tmp_path / 'task-eval', [{}]
    evaluator.prompt_rendering = 'full'
    counts = iter([4, 6, 5, 3])
    calls = []

    def execute(run, repeat=0):
        run.mkdir(parents=True, exist_ok=True)
        calls.append(run.name)
        n = next(counts)
        return {'task_successes': n, 'task_success_rate': n / 16, 'complete': True, 'eligible_for_best': True,
                'cases': [], 'cases_expected': 16,
                'outcome_counts': {'task_success': n, 'infrastructure_error': 0, 'unresolved': 0,
                                   'semantic_mismatch': 16 - n}}, [0]
    evaluator._execute = execute
    first = evaluator.run(10)
    assert calls == ['step-0000010', 'step-0000010-repeat1', 'step-0000010-repeat2']
    assert first['repeat_noise'] == 2 and first['noise_successes'] == 2 and first['task_successes'] == 4
    second = evaluator.run(20)  # 3 successes: within noise of 4, no repeat
    assert calls[-1] == 'step-0000020' and second['repeat_noise'] is None
    assert second['noise_successes'] == 2 and second['previous_successes'] == 4 and second['regressed'] is False


def test_sampled_evaluations_send_their_temperature_and_repeat_runs_draw_new_seeds(tmp_path):
    """REPETITION_DRAGNET_20261013.md: at the default temperature 0 the server decodes greedily, so a repeat run
    replays the same actions; a declared temperature reaches every queue entry and each repeat gets its own seeds."""
    from natlang_neuralese.eval.task_cadence import TaskEval

    def queues(**extra):
        evaluator = TaskEval.__new__(TaskEval)
        evaluator.spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [3, 7], **extra})
        evaluator.rows, evaluator.prompt_rendering = [{}, {}], 'program-header'
        evaluator.subset = tmp_path / 'subset.jsonl'
        return [[(e.get('temperature'), e['seed']) for e in entries]
                for repeat in (0, 1) for _, _, entries in evaluator._queues(tmp_path / f'r{repeat}', repeat)]
    assert queues() == [[(None, 400003)], [(None, 400007)], [(None, 400003)], [(None, 400007)]]
    assert queues(temperature=0.6) == [[(0.6, 400003)], [(0.6, 400007)], [(0.6, 500003)], [(0.6, 500007)]]
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'temperature': 3})


def test_budget_exhaustion_is_a_model_outcome_and_infrastructure_errors_are_not():
    from natlang_neuralese.eval.task_cadence import accounted
    base = {'task_success': 1, 'budget_exhaustion': 3, 'semantic_mismatch': 12, 'infrastructure_error': 0,
            'unresolved': 0}
    assert accounted({'cases_expected': 16, 'outcome_counts': base})
    assert not accounted({'cases_expected': 16, 'outcome_counts': {**base, 'semantic_mismatch': 11,
                                                                    'infrastructure_error': 1}})
    assert not accounted({'cases_expected': 16, 'outcome_counts': {**base, 'semantic_mismatch': 11, 'unresolved': 1}})


def test_a_failed_evaluation_is_recorded_and_retried_and_repeated_failures_stop_the_run(tmp_path):
    from natlang_neuralese.eval.task_cadence import TaskEval
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_repeats': 1,
                            'max_consecutive_failures': 2})
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out, evaluator.root, evaluator.rows = spec, tmp_path, tmp_path / 'task-eval', [{}]
    evaluator.prompt_rendering = 'full'
    outcomes = iter(['crash', 'infra', 'ok', 'crash', 'infra'])

    def execute(run, repeat=0):
        run.mkdir(parents=True, exist_ok=True)
        kind = next(outcomes)
        if kind == 'crash':
            raise OSError('collector runtime missing while ts-host rebuilds')
        infra = int(kind == 'infra')
        return {'task_successes': 1 - infra, 'task_success_rate': 0.0, 'complete': True, 'cases': [],
                'cases_expected': 1, 'outcome_counts': {'task_success': 1 - infra, 'infrastructure_error': infra,
                                                        'unresolved': 0}}, [0]
    evaluator._execute = execute
    first = evaluator.cadence(10)  # an exception: recorded, the run continues
    assert first['eligible'] is False and 'OSError' in first['error']
    with pytest.raises(RuntimeError, match='2 times in a row'):
        evaluator.cadence(20)  # a second ineligible evaluation in a row stops the run
    assert evaluator.cadence(30)['eligible'] is True  # a resumed run evaluates again; success resets the count
    assert evaluator.consecutive_failures() == 0
    assert evaluator.cadence(40)['eligible'] is False
    with pytest.raises(RuntimeError):
        evaluator.cadence(50)
    import json
    rows = [json.loads(line) for line in (tmp_path / 'task-eval.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [10, 20, 30, 40, 50]


def _light_evaluator(spec, out):
    from natlang_neuralese.eval.task_cadence import TaskEval
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out, evaluator.root, evaluator.rows = spec, out, out / 'task-eval', [{}]
    evaluator.prompt_rendering, evaluator.engine = 'full', None
    evaluator.root.mkdir(parents=True, exist_ok=True)
    return evaluator


def test_an_async_evaluation_is_submitted_from_a_weights_only_snapshot_and_lands_later(tmp_path, monkeypatch):
    """Coordinator 2026-10-13: the trainer links its just-written slot, a child prunes it to weights only, training
    goes on; the job restores the weights, appends the row with its step and deletes the snapshot."""
    import json
    import torch
    from natlang_neuralese.eval import task_cadence
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_repeats': 1,
                            'async': True})
    evaluator = _light_evaluator(spec, tmp_path)
    monkeypatch.setattr(task_cadence, 'request_queue', lambda: tmp_path / 'queue')
    slot = tmp_path / 'checkpoint.pt'
    torch.save({'schema': 'natlang.neuralese-direct/1', 'step': 40,
                'student_parameters': {'backbone.w': torch.ones(2)},
                'optimizer': {'state': {'m': torch.zeros(2)}}}, slot)
    pending = evaluator.submit(40, slot, heads='heads.pt', device='cpu')
    assert pending['pending'] is True and pending['eligible'] is None
    request_dir = tmp_path / 'task-eval' / 'async' / 'step-0000040'
    for thread in __import__('threading').enumerate():
        if thread.name.startswith('task-eval-prune'):
            thread.join(60)
    weights = torch.load(request_dir / 'weights.pt', weights_only=False)
    assert 'optimizer' not in weights and weights['optimizer_pruned'] is True
    assert not (request_dir / 'full.pt').exists()
    torch.save({'step': 41}, tmp_path / 'next.pt')
    (tmp_path / 'next.pt').replace(slot)  # the next rolling write never touches the snapshot
    assert task_cadence.pending_requests(tmp_path) == [request_dir]
    assert [json.loads(p.read_text())['request'] for p in (tmp_path / 'queue').iterdir()] == [str(request_dir)]

    monkeypatch.setattr(task_cadence.TaskEval, '__init__',
                        lambda self, spec, out, engine, prompt_rendering:
                        self.__dict__.update(_light_evaluator(spec, out).__dict__))
    seen = {}

    def execute(self, run):
        run.mkdir(parents=True, exist_ok=True)
        seen['engine'] = self.engine
        return {'task_successes': 3, 'task_success_rate': 3 / 16, 'complete': True, 'cases': [],
                'cases_expected': 1, 'outcome_counts': {'task_success': 1, 'infrastructure_error': 0,
                                                        'unresolved': 0}}, [0]
    monkeypatch.setattr(task_cadence.TaskEval, '_execute', execute)
    summary = task_cadence.run_request(request_dir, engine_factory=lambda request: ('engine', request['heads']))
    assert seen['engine'] == ('engine', 'heads.pt') and summary['step'] == 40 and summary['eligible'] is True
    assert not request_dir.exists() and task_cadence.pending_requests(tmp_path) == []
    rows = [json.loads(line) for line in (tmp_path / 'task-eval.jsonl').read_text().splitlines()]
    assert [(r['step'], r['task_successes']) for r in rows] == [(40, 3)]
    # A requeued job after the row landed only cleans up.
    request_dir.mkdir(parents=True)
    (request_dir / 'request.json').write_text(json.dumps({'step': 40, 'spec': spec, 'stage_out': str(tmp_path),
                                                          'prompt_rendering': 'full'}))
    assert task_cadence.run_request(request_dir)['task_successes'] == 3 and not request_dir.exists()


def test_a_failed_async_job_is_an_ineligible_row_and_pending_ones_are_no_failure(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.eval import task_cadence
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'async': True,
                            'max_consecutive_failures': 2})
    evaluator = _light_evaluator(spec, tmp_path)
    monkeypatch.setattr(task_cadence, 'request_queue', lambda: tmp_path / 'queue')
    request_dir = tmp_path / 'task-eval' / 'async' / 'step-0000010'
    request_dir.mkdir(parents=True)
    (request_dir / 'request.json').write_text(json.dumps({'step': 10, 'spec': spec, 'stage_out': str(tmp_path),
                                                          'prompt_rendering': 'full'}))
    monkeypatch.setattr(task_cadence.TaskEval, '__init__',
                        lambda self, spec, out, engine, prompt_rendering:
                        self.__dict__.update(_light_evaluator(spec, out).__dict__))
    summary = task_cadence.run_request(request_dir)  # no snapshot: recorded, not raised
    assert summary['eligible'] is False and 'FileNotFoundError' in summary['error'] and not request_dir.exists()
    assert evaluator.consecutive_failures() == 1
    slot = tmp_path / 'checkpoint.pt'
    slot.write_bytes(b'x')
    monkeypatch.setattr(task_cadence, 'prune_snapshot', lambda d: None)
    monkeypatch.setattr(task_cadence.threading, 'Thread', lambda **k: type('T', (), {'start': lambda s: None})())
    assert evaluator.submit(20, slot, heads=None)['pending'] is True  # one landed failure, one pending: continues
    evaluator.record_failure(20, 'job ended without a result')
    with pytest.raises(RuntimeError, match='2 times in a row'):
        evaluator.submit(30, slot, heads=None)


def test_handoff_readiness_waits_for_the_candidate_checkpoints_task_evaluation(tmp_path):
    import json
    from natlang_neuralese.train.e2e_acceptance import _task_eval_candidate
    assert _task_eval_candidate(tmp_path) is None
    row = {'step': 400, 'eligible': True, 'task_successes': 8, 'cases': 16, 'reference_successes': 4}
    (tmp_path / 'task-eval.jsonl').write_text(json.dumps(row) + '\n')
    assert _task_eval_candidate(tmp_path)['rate'] == 0.5
    (tmp_path / 'task-eval' / 'async' / 'step-0000800').mkdir(parents=True)
    (tmp_path / 'task-eval' / 'async' / 'step-0000800' / 'request.json').write_text('{}')
    waiting = _task_eval_candidate(tmp_path)
    assert waiting['status'] == 'pending' and waiting['step'] == 800


def test_the_dispatcher_finds_requests_through_the_queue_pointers(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'task_eval_dispatch', Path(__file__).resolve().parents[2] / 'scripts' / 'task_eval_dispatch.py')
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)
    queue, request_dir = tmp_path / 'queue', tmp_path / 'stage' / 'task-eval' / 'async' / 'step-0000400'
    request_dir.mkdir(parents=True)
    (request_dir / 'request.json').write_text('{}')
    queue.mkdir()
    (queue / 'a-step-0000400.json').write_text(json.dumps({'request': str(request_dir)}))
    (queue / 'b-step-0000800.json').write_text(json.dumps({'request': str(tmp_path / 'gone')}))
    assert dispatch.requests(queue) == [request_dir]
    assert sorted(p.name for p in queue.iterdir()) == ['a-step-0000400.json']  # landed requests drop their pointer
    (request_dir / 'weights.pt').write_bytes(b'\0' * 2**20)
    assert dispatch.budget_gb(request_dir) == round(2**20 / 2**30 * 1.3 + 8, 1)


def test_pending_requests_are_capped_and_a_full_mirror_skips_without_failing(tmp_path, monkeypatch):
    """Coordinator 2026-10-13 (ENOSPC lesson): at most max_pending unlanded snapshots per stage, older ones dropped
    as superseded; no snapshot below the mirror's floor. Neither counts as a failed evaluation."""
    import json
    from natlang_neuralese.eval import task_cadence
    from natlang_neuralese.train import checkpoint_offload
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'async': True,
                            'max_consecutive_failures': 1})
    evaluator = _light_evaluator(spec, tmp_path)
    monkeypatch.setattr(task_cadence, 'request_queue', lambda: tmp_path / 'queue')
    monkeypatch.setattr(task_cadence.threading, 'Thread', lambda **k: type('T', (), {'start': lambda s: None})())
    slot = tmp_path / 'checkpoint.pt'
    slot.write_bytes(b'x' * 1000)
    for step in (100, 200, 300):
        evaluator.submit(step, slot, heads=None)  # max_consecutive_failures 1: a superseded row must not raise
    assert [p.name for p in task_cadence.pending_requests(tmp_path)] == ['step-0000200', 'step-0000300']
    rows = [json.loads(line) for line in (tmp_path / 'task-eval.jsonl').read_text().splitlines()]
    assert [(r['step'], r['superseded']) for r in rows] == [(100, True)] and evaluator.consecutive_failures() == 0
    floor = checkpoint_offload.OffloadSettings(hdd_floor_bytes=10**18)
    monkeypatch.setattr(checkpoint_offload.OffloadSettings, 'from_env', classmethod(lambda cls: floor))
    request_dir = tmp_path / 'task-eval' / 'async' / 'step-0000300'
    evaluator._prune_in_background(request_dir)
    assert not request_dir.exists()
    last = json.loads((tmp_path / 'task-eval.jsonl').read_text().splitlines()[-1])
    assert last['step'] == 300 and last['skipped'] is True and 'floor' in last['error']
    assert evaluator.consecutive_failures() == 0
