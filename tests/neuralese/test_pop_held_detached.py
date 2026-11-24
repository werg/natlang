"""Pop held evaluations: the checkpoint prefetch, the detached evaluation job (pop_rl_actor held-eval) and grpo's
adoption of a running one after a restart."""
import json
import subprocess
import sys

import pytest

sys.path.insert(0, 'scripts/neuralese_data')
import grpo  # noqa: E402
import pop_rl_actor  # noqa: E402


@pytest.fixture
def pop(tmp_path, monkeypatch):
    """A fake Pop: ``files`` on its disk, the ssh commands it ran and the rsyncs sent to it."""
    state = {'files': set(), 'ssh': [], 'copies': []}

    def ssh(command, **_):
        state['ssh'].append(command)
        if command.startswith('test -f'):
            return 'ok' if command.split()[2] in state['files'] else ''
        if command.startswith('mv'):
            src, dst = command.split()[1:3]
            state['files'].discard(src)
            state['files'].add(dst)
        if command.startswith('df'):
            return 'Avail\n' + str(state.get('free', 100 * 2**30))
        return ''

    def copy(source, dest, **_):
        state['copies'].append(dest)
        state['files'].add(dest.split(':', 1)[1])

    monkeypatch.setattr(pop_rl_actor, 'ssh', ssh)
    monkeypatch.setattr(pop_rl_actor, 'copy_resumable', copy)
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path / 'root')
    monkeypatch.setattr(pop_rl_actor, 'STATE', tmp_path / 'root/rl-actor.json')
    monkeypatch.setattr(pop_rl_actor, 'POP', 'pop')
    monkeypatch.setattr(pop_rl_actor, 'prune_pop_newest', lambda *a, **k: state.setdefault('pruned', []).append((a, k)))
    return state


def test_checkpoint_goes_to_pop_once(pop, tmp_path):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'x')
    assert pop_rl_actor.checkpoint_on_pop(checkpoint, 'a' * 64) is True
    assert pop['copies'] == [f'pop:{pop_rl_actor.POP_CHECKPOINTS}/{"a" * 64}.pt.partial']
    assert pop_rl_actor.checkpoint_on_pop(checkpoint, 'a' * 64) is False   # present: serve-checkpoint skips the copy
    assert len(pop['copies']) == 1


def test_prefetch_keeps_the_served_checkpoint_and_refuses_a_full_disk(pop, tmp_path):
    from types import SimpleNamespace
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'x' * 10)
    (tmp_path / 'root').mkdir()
    (tmp_path / 'root/rl-actor.json').write_text(json.dumps({'policy_id': 'b' * 64}))
    args = SimpleNamespace(checkpoint=str(checkpoint), policy_id='a' * 64, step=6)
    assert pop_rl_actor.prefetch(args)['event'] == 'Pop checkpoint prefetched'
    (pattern, keep), kwargs = pop['pruned'][0]
    assert keep == pop_rl_actor.POP_KEEP + 1 and any('b' * 64 in path for path in kwargs['protect'])
    pop['free'] = 2**30   # under the scratch floor
    with pytest.raises(RuntimeError, match='no prefetch'):
        pop_rl_actor.prefetch(SimpleNamespace(**{**vars(args), 'policy_id': 'c' * 64}))


def _spec(tmp_path, base_complete=True):
    base, out = tmp_path / 'base', tmp_path / 'step-00004'
    base.mkdir()
    out.mkdir()
    (base / 'result.json').write_text(json.dumps({'complete': base_complete}))
    spec = {'unit': 'u', 'policy': 'p' * 64, 'format': 'fmt', 'serve': ['serve-checkpoint', '--policy-id', 'p' * 64],
            'measure': ['--out', str(out)], 'slots': 20, 'log': str(tmp_path / 'held-pop.log'),
            'compare': [str(base / 'result.json'), str(out / 'result.json'), '--out', str(out / 'compare-vs-step0.json')],
            'done': str(out / 'pop-eval-done.json')}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    return spec, out


@pytest.fixture(autouse=True)
def keep_sigterm(monkeypatch):
    monkeypatch.setattr(pop_rl_actor.signal, 'signal', lambda *_a: None)   # the job's handler stays out of pytest


def test_held_eval_job_serves_measures_and_compares(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec, out = _spec(tmp_path)
    served, calls = [], []
    monkeypatch.setattr(pop_rl_actor, 'serving_endpoint', lambda policy, fmt: 'http://pop:1' if served else None)

    def run(argv, **_):
        calls.append(argv)
        if 'serve-checkpoint' in argv:
            served.append(True)
        if 'measure' in argv:
            assert argv[-2:] == ['--endpoint', 'http://pop:1=20']
            (out / 'result.json').write_text(json.dumps({'complete': True}))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', run)
    assert pop_rl_actor.held_eval_job(SimpleNamespace(spec=str(tmp_path / 'spec.json'))) == 0
    assert 'serve-checkpoint' in calls[0] and [c[3] for c in calls[1:]] == ['measure', 'compare']
    done = json.loads((out / 'pop-eval-done.json').read_text())
    assert done['returncode'] == 0 and done['served'] and done['policy'] == 'p' * 64


def test_held_eval_job_skips_compare_of_a_cut_measurement_and_records_a_failed_serve(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec, out = _spec(tmp_path)
    (out / 'compare-vs-step0.json').write_text('{}')   # stale
    calls = []
    monkeypatch.setattr(pop_rl_actor, 'serving_endpoint', lambda policy, fmt: 'http://pop:1')

    def run(argv, **_):
        calls.append(argv)
        (out / 'result.json').write_text(json.dumps({'complete': False}))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', run)
    pop_rl_actor.held_eval_job(SimpleNamespace(spec=str(tmp_path / 'spec.json')))
    assert [c[3] for c in calls] == ['measure'] and not (out / 'compare-vs-step0.json').exists()

    monkeypatch.setattr(pop_rl_actor, 'serving_endpoint', lambda policy, fmt: None)   # serve never attests
    assert pop_rl_actor.held_eval_job(SimpleNamespace(spec=str(tmp_path / 'spec.json'))) == 1
    done = json.loads((out / 'pop-eval-done.json').read_text())
    assert done['served'] is False and done['returncode'] == 1
    assert 'Pop does not serve' in (tmp_path / 'held-pop.log').read_text()


def test_a_restarted_run_adopts_only_a_running_unfinished_evaluation(tmp_path):
    a, b, c = (tmp_path / f'step-0000{i}' for i in (2, 4, 6))
    for directory, unit in ((a, 'ua'), (b, 'ub'), (c, 'uc')):
        directory.mkdir()
        (directory / 'pop-eval.json').write_text(json.dumps({'unit': unit, 'policy': 'p', 'step': 1, 'started': 0}))
    (a / 'pop-eval-done.json').write_text('{}')            # finished: its outcome is there
    active = {'ua', 'uc'}.__contains__                      # ub's unit is gone (killed): resumed as a partial instead
    assert grpo.pop_eval_inflight([a, b, c], active) == (c, json.loads((c / 'pop-eval.json').read_text()))
    assert grpo.pop_eval_inflight([a, b], active) is None


def test_held_priors_are_every_complete_result_but_the_measured_one(tmp_path):
    base, held = tmp_path / 'base', tmp_path / 'held'
    for directory, complete in ((base, True), (held / 'step-00002', True), (held / 'step-00004', False),
                                (held / 'step-00006', True)):
        directory.mkdir(parents=True)
        (directory / 'result.json').write_text(json.dumps({'complete': complete}))
    assert grpo.held_priors(base, held, held / 'step-00006') == [str(base / 'result.json'),
                                                                  str(held / 'step-00002/result.json')]


def test_start_and_serve_checkpoint_claim_one_admissible_pop_budget(monkeypatch):
    # 10.5 GB was refused beside Pop's gdm greeter (10.8 GiB free, 2026-11-20); both actor paths claim the same budget
    import inspect
    monkeypatch.delenv('NATLANG_POP_HELD_BUDGET_GB', raising=False)
    # 10.2 GiB admissible beside other agents' resident Pop processes (2026-11-21): 10 + 0.5 timed out twice
    assert float(pop_rl_actor.pop_sglang_budget_gb()) + pop_rl_actor.POP_RESERVE_GB <= 10.2
    monkeypatch.setenv('NATLANG_POP_HELD_BUDGET_GB', '9.0')
    assert pop_rl_actor.pop_sglang_budget_gb() == '9.0'
    for path in (pop_rl_actor.start, pop_rl_actor._serve_pop):     # serve_checkpoint serves Pop through _serve_pop
        assert 'pop_sglang_budget_gb()' in inspect.getsource(path)


def test_held_eval_job_refuses_the_superseded_colab_series(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec, out = _spec(tmp_path)
    spec['host_profile'] = 'colab_g4'
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', lambda *a, **k: pytest.fail('nothing runs'))
    assert pop_rl_actor.held_eval_job(SimpleNamespace(spec=str(tmp_path / 'spec.json'))) == 1
    assert 'superseded by the DGX series' in (tmp_path / 'held-pop.log').read_text()
