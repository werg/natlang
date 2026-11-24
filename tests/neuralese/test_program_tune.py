"""Per-program LoRA tuning driver (scripts/neuralese_data/program_tune.py): split, verified turn rows, the plan, and that
its rows lower and fit through the tuner on a tiny native engine."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))

import program_tune as P  # noqa: E402
from test_native_serving_adapters import engine_fixture  # noqa: E402

TOOLS = [{'type': 'function', 'function': {'name': 'eval', 'description': 'Run code.',
                                           'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}}}},
         {'type': 'function', 'function': {'name': 'return_result', 'description': 'Return.',
                                           'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}}]


def call(ident, name, arguments):
    return {'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}


def episode():
    """eval (error) -> eval (ok) -> return_result: the first action failed, the later ones are verified (reply texts are
    toy-vocabulary words: the fixture's template renders content, not tool calls)."""
    system, user = {'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 't4 t5'}
    a1 = {'role': 'assistant', 'content': '', 'tool_calls': [call('c1', 'eval', {'code': 'x('})]}
    r1 = {'role': 'tool', 'tool_call_id': 'c1', 'content': '1:2 typescript-syntax: Expression expected.'}
    a2 = {'role': 'assistant', 'content': 't6', 'reasoning_content': 'retry',
          'tool_calls': [call('c2', 'eval', {'code': 'x()'})]}
    r2 = {'role': 'tool', 'tool_call_id': 'c2', 'content': 'console: 7'}
    a3 = {'role': 'assistant', 'content': 't7', 'tool_calls': [call('c3', 'return_result', {'value': 't7'})]}
    return [{'messages': [system, user], 'tools': TOOLS, 'target': a1, 'invocation_id': 'inv'},
            {'messages': [system, user, a1, r1], 'tools': TOOLS, 'target': a2, 'invocation_id': 'inv'},
            {'messages': [system, user, a1, r1, a2, r2], 'tools': TOOLS, 'target': a3, 'invocation_id': 'inv'}]


def test_split_is_stable_and_covers_the_fractions():
    names = [P.split_of('fam', f'g{i}') for i in range(4000)]
    assert names == [P.split_of('fam', f'g{i}') for i in range(4000)]
    shares = {name: names.count(name) / len(names) for name in P.SPLITS}
    assert abs(shares['support'] - .6) < .03 and abs(shares['dev'] - .25) < .03 and abs(shares['confirm'] - .15) < .03
    assert P.split_of('other', 'g1') in P.SPLITS


def test_error_results_are_recognised():
    for text in ('1:7 invalid-binding: messages is this call\'s input', 'ReferenceError: x is not defined',
                 'Error: move m1 was already made', 'rejected: the value is not a string',
                 'intent: failed: decision collection needs its exact options', 'TypeError: y'):
        assert P.ERROR_RESULT.match(text), text
    for text in ('console: 3', 'made m1', 'request: string = "x"', 'Natural-language function source', '[]'):
        assert not P.ERROR_RESULT.match(text), text


def test_failed_actions_are_not_targets_and_mark_later_history():
    rows, dispositions = P.episode_turns('task-1', 'rl:run:g0:0', episode())
    assert dispositions == ['rejected', 'admissible', 'admissible']
    assert [row['turn'] for row in rows] == [1, 2]
    assert all(row['evidence']['failedHistory'] for row in rows)
    assert rows[0]['reply'] == {'text': 't6', 'calls': [['eval', {'code': 'x()'}]], 'reasoning': 'retry'}
    assert rows[1]['evidence']['actionEvents'][0]['outcome'] == 'ok'   # the final reply: the oracle verdict


def test_a_call_without_a_recorded_result_is_unverified_unless_final():
    turns = episode()[:2]
    _, dispositions = P.episode_turns('task-1', 'k', turns)
    assert dispositions == ['rejected', 'admissible']   # the last reply ends the episode
    turns[1]['messages'] = turns[1]['messages'][:2]   # the first call's result was never recorded
    turns.append({**turns[1], 'target': turns[1]['target']})
    _, dispositions = P.episode_turns('task-1', 'k', turns)
    assert dispositions[0] == 'unverified'


def test_a_nested_calls_last_turn_is_judged_by_the_episode():
    turns = episode()[:2]
    turns[1]['messages'] = turns[1]['messages'][:2]   # no later turn shows the first reply's result ...
    nested = {**turns[0], 'invocation_id': 'child'}   # ... because it was a nested call's last turn
    root = {**turns[1], 'invocation_id': 'root'}
    _, dispositions = P.episode_turns('task-1', 'k', [nested, root])
    assert dispositions[0] != 'unverified'


def test_rows_lower_and_fit_through_the_tuner(tmp_path):
    from natlang_neuralese.tune.rows import lower_rows
    from natlang_neuralese.tune.trainer import JointTrainer, Recipe
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    rows, _ = P.episode_turns('task-1', 'rl:run:g0:0', episode())
    lowered = lower_rows(engine, rows, prompt_weight=0.)
    assert all(row['binding'] == P.BINDING for row in lowered)
    node = lowered[0]['graph']['nodes'][0]
    assert 0 < sum(node['target_weights']) < len(node['token_ids'])   # the reply only
    manifest = P.manifest('fam', {'checkpoint_sha256': engine.native_identity['checkpoint_sha256']})
    recipe = Recipe.from_json({'parameterization': 'free-lora', 'ranks': {'tree': 2, 'folder': 0, 'lambda': 0},
                               'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.}, 'steps': 2, 'batch_rows': 2})
    trainer = JointTrainer(engine, manifest, lowered, recipe)
    first = trainer.step()
    second = trainer.step()
    assert set(trainer.state['node_artifacts']) == {'tree:'}
    assert sum(second['row_losses']) < sum(first['row_losses'])


def test_a_served_release_is_the_default_stack_and_a_distinct_policy(tmp_path):
    import torch
    from natlang_neuralese.model.shared_state import Invocation, compile_graph
    from natlang_neuralese.rl.held_eval import policy_id_of
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.tune.build import persist_artifacts
    from natlang_neuralese.tune.rows import lower_rows
    from natlang_neuralese.tune.trainer import JointTrainer, Recipe
    from serve_native_checkpoint import bind_adapter_release
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    rows, _ = P.episode_turns('task-1', 'rl:run:g0:0', episode())
    recipe = Recipe.from_json({'parameterization': 'free-lora', 'ranks': {'tree': 2, 'folder': 0, 'lambda': 0},
                               'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.}, 'steps': 3, 'batch_rows': 2,
                               'optimizer': {'name': 'adam', 'learning_rate': 5e-2}})
    trainer = JointTrainer(engine, P.manifest('fam', {}), lower_rows(engine, rows, prompt_weight=0.), recipe)
    for _ in range(3):
        trainer.step()
    release = tmp_path / 'release'
    persist_artifacts(engine, list(trainer.state['node_artifacts'].values()), release / 'artifacts', release / 'states')
    (release / 'result.json').write_text(json.dumps(trainer.manifest_fragment()))
    served = load_native_engine(str(tmp_path / 'native.pt'), device=engine.device)
    base_id = policy_id_of(served.rl_policy_binding)
    entries = bind_adapter_release(served, release)
    assert served.native_default_adapters == entries == [{'id': trainer.state['node_artifacts']['tree:'], 'scale': 1.0}]
    adapted_id = policy_id_of(served.rl_policy_binding)
    assert adapted_id.startswith(base_id + '+') and adapted_id != base_id
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    model = served.shared_model
    run = lambda adapters: served._native_run(lambda: model.logits(model(plan)['h_final']).detach(), adapters)
    assert not torch.equal(run(None), run([]))   # the default stack applies; an explicit empty stack is the base


def toy_program(engine, root, family, tasks, seed):
    """A program directory with `tasks` one-turn support successes and its own 2-step free-LoRA fit."""
    from natlang_neuralese.tune.build import persist_artifacts
    from natlang_neuralese.tune.rows import lower_rows
    from natlang_neuralese.tune.trainer import JointTrainer, Recipe
    directory = root / family
    directory.mkdir(parents=True)
    ids = [f'{family}:task-{i}' for i in range(tasks)]
    (directory / 'split.json').write_text(json.dumps({'family': family, 'ids': {'support': ids, 'dev': [], 'confirm': []}}))
    turns = []
    for i, task in enumerate(ids):
        target = {'role': 'assistant', 'content': f't{5 + i % 5} t{5 + (i // 5) % 5} t{5 + seed % 5}'}
        made, _ = P.episode_turns(task, f'sample:{task}', [{'messages': [{'role': 'system', 'content': 'sys'},
                                  {'role': 'user', 'content': f't4 t{5 + i % 5}'}], 'tools': [], 'target': target}])
        turns += made
    (directory / 'rows.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in turns))
    recipe = Recipe.from_json({'parameterization': 'free-lora', 'ranks': {'tree': 2, 'folder': 0, 'lambda': 0},
                               'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.}, 'steps': 2, 'batch_rows': 3,
                               'seed': seed, 'optimizer': {'name': 'adam', 'learning_rate': 5e-2}})
    trainer = JointTrainer(engine, P.manifest(family, {}), lower_rows(engine, turns, prompt_weight=0.), recipe)
    trainer.step(); trainer.step()
    build = directory / 'fit' / 'build'
    persist_artifacts(engine, list(trainer.state['node_artifacts'].values()), build / 'artifacts', build / 'states')
    (build / 'result.json').write_text(json.dumps(trainer.manifest_fragment()))
    return directory


def test_bank_from_program_fits_adapts_a_held_program_into_a_servable_release(tmp_path):
    import argparse
    from natlang_neuralese.serve import load_native_engine
    from serve_native_checkpoint import bind_adapter_release
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    programs = [toy_program(engine, tmp_path / 'programs', f'fam{k}', 6, k) for k in range(3)]
    held = toy_program(engine, tmp_path / 'programs', 'held', 7, 9)
    recipe = {'rank': 4, 'components': 4, 'component_rank': 2, 'inner_steps': 2, 'layer_groups': 1, 'meta_batch': 1,
              'aux_families': 1, 'init_families': 3, 'lora_rank': 2, 'reference_passes': [1]}
    args = argparse.Namespace(programs=[str(p) for p in programs], held=[str(held)], out=str(tmp_path / 'bank'),
                              checkpoint=str(tmp_path / 'native.pt'), checkpoint_sha256=None, base=None, device=engine.device,
                              dtype=None, store_scratch=None, steps=1, rows_per_world=8, seed=0, recipe=json.dumps(recipe),
                              target_pattern=r'self_attn\.(q_proj|k_proj|v_proj|out_proj)|conv\.(in_proj|out_proj)',
                              max_worlds=6, held_support_rows=4)
    P.bank(args)
    bank = tmp_path / 'bank'
    pool = json.loads((bank / 'pool' / 'pool.json').read_text())
    assert {name: f['role'] for name, f in pool['families'].items()} == {'fam0': 'meta-train', 'fam1': 'meta-train',
                                                                         'fam2': 'meta-train', 'held': 'held'}
    assert json.loads((bank / 'learner' / 'state.json').read_text())['step'] == 1
    assert (bank / 'bank-manifest.json').exists()
    assert json.loads((bank / 'adapt' / 'held' / 'result.json').read_text())['receipt']['support_rows'] == 4
    served = load_native_engine(str(tmp_path / 'native.pt'), device=engine.device)
    entries = bind_adapter_release(served, bank / 'adapt' / 'held')
    assert served.rl_policy_binding['adapters']['stack'] == entries


def test_a_differentiated_adapter_row_leaves_no_cyclic_garbage(tmp_path):
    """Each fit row's effective weights are freed when its operation ends, not whenever Python's cyclic collector runs
    (native_actor_graph.release_operation_cycles; on LFM3 the cycles held ~2.5 GB per row until a step ended)."""
    import gc
    from natlang_neuralese.tune.rows import lower_rows
    from natlang_neuralese.tune.trainer import JointTrainer, Recipe
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    rows, _ = P.episode_turns('task-1', 'k', episode())
    recipe = Recipe.from_json({'parameterization': 'free-lora', 'ranks': {'tree': 2, 'folder': 0, 'lambda': 0},
                               'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.}, 'steps': 2, 'batch_rows': 2})
    trainer = JointTrainer(engine, P.manifest('fam', {}), lower_rows(engine, rows, prompt_weight=0.), recipe)
    # The first step also leaves one-time garbage of lazy imports and caches (667 objects when this test ran first,
    # 2026-11-24): measure a row's operation on the second step.
    trainer.step()
    gc.collect()
    gc.disable()
    try:
        trainer.step()
        left = gc.collect()
    finally:
        gc.enable()
    assert left < 50, left


def test_pop_actor_serves_an_adapter_release_on_sglang_slots():
    import pop_actor_driver
    assert pop_actor_driver.adapter_args({}) == ['--sglang-adapter-slots', '0']
    job = {'adapter_release': {'path': '/eval/adapter-ab', 'rank': 16, 'scale': 1.0, 'result_sha256': 'ab'}}
    assert pop_actor_driver.adapter_args(job) == ['--adapter-release', '/eval/adapter-ab', '--adapter-scale', '1.0',
                                                  '--sglang-adapter-slots', '1']   # the server derives the slot
    job['adapter_release']['slot_rank'] = 48   # pop_rl_actor.send_adapter_release: the packed rank, not the fit's 16
    assert pop_actor_driver.adapter_args(job)[-2:] == ['--sglang-adapter-rank', '48']


def test_sglang_slot_rank_fits_a_release_over_every_lfm3_family(monkeypatch, tmp_path):
    """A rank-16 fit over every LFM3 family packs q|k|v into one fork module (48) and w1|w3 into another (32): the
    server sizes its slots from the release, so the default 32 no longer sends every adapted turn to the native engine."""
    import serve_native_checkpoint as S
    from natlang_neuralese.export.native_hf import packed_slot_rank
    layer = lambda i, family: f'port.hf.model.layers.{i}.{family}.weight'
    attention = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.gate_proj']
    conv = ['conv.in_proj', 'conv.out_proj']
    mlp = ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
    names = [layer(2, f) for f in attention + mlp] + [layer(0, f) for f in conv + mlp]
    fit = {'kind': 'free-lora', 'rank': 16, 'parameter_names': names}
    assert packed_slot_rank([fit]) == 48
    assert packed_slot_rank([fit, {'kind': 'xs', 'rank': 4, 'parameter_names': [layer(2, 'self_attn.k_proj')]}]) == 52
    assert packed_slot_rank([{'kind': 'free-lora', 'rank': 16, 'parameter_names': [layer(0, f) for f in mlp]}]) == 32
    with pytest.raises(ValueError):
        packed_slot_rank([{'kind': 'free-lora', 'rank': 4, 'parameter_names': ['port.hf.model.embed_tokens.weight']}])
    args = S.build_parser().parse_args(['--checkpoint', 'c', '--checkpoint-sha256', 'x'])
    assert args.sglang_adapter_rank is None and S.adapter_slot_rank(args) == S.DEFAULT_ADAPTER_RANK
    monkeypatch.setattr(S, 'release_slot_rank', lambda release: 48)
    args = S.build_parser().parse_args(['--checkpoint', 'c', '--checkpoint-sha256', 'x', '--adapter-release', str(tmp_path)])
    assert S.adapter_slot_rank(args) == 48
    args = S.build_parser().parse_args(['--checkpoint', 'c', '--checkpoint-sha256', 'x', '--adapter-release', str(tmp_path),
                                        '--sglang-adapter-rank', '16'])
    assert S.adapter_slot_rank(args) == 16   # explicit wins (adapted turns over it run on the native engine)


def test_pop_adapter_job_carries_the_packed_slot_rank(monkeypatch, tmp_path):
    """The DGX records the slot the release needs on SGLang (packed q|k|v), not the fit's rank, in the Pop job."""
    import pop_rl_actor
    import serve_native_checkpoint as S
    monkeypatch.setattr(pop_rl_actor, 'ssh', lambda *a, **k: '')
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(S, 'release_slot_rank', lambda release: 48)
    job = {}
    pop_rl_actor.send_adapter_release(job, '/eval', tmp_path, 'ab' * 8, 16)
    assert job['adapter_release']['rank'] == 16 and job['adapter_release']['slot_rank'] == 48


def test_pop_attestation_distinguishes_base_and_adapted_actors(monkeypatch):
    import io
    import pop_rl_actor
    def info(adapters):
        binding = {'checkpoint_sha256': 'c' * 64, 'actor_export': {'format': 'x'},
                   **({'adapters': {'result_sha256': adapters}} if adapters else {})}
        return lambda *_a, **_k: io.BytesIO(json.dumps({'rl_policy_binding': binding}).encode())
    monkeypatch.setattr(pop_rl_actor.urllib.request, 'urlopen', info(None))
    assert pop_rl_actor.attested('http://pop', 'c' * 64) and not pop_rl_actor.attested('http://pop', 'c' * 64, 'ab')
    monkeypatch.setattr(pop_rl_actor.urllib.request, 'urlopen', info('ab'))
    assert pop_rl_actor.attested('http://pop', 'c' * 64, 'ab') and not pop_rl_actor.attested('http://pop', 'c' * 64)


def test_release_keeps_the_result_and_only_its_artifact_closure(tmp_path):
    import argparse
    from natlang_neuralese.serve import load_native_engine
    from serve_native_checkpoint import bind_adapter_release
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    directory = toy_program(engine, tmp_path / 'programs', 'fam', 4, 1)
    build = directory / 'fit' / 'build'
    (build / 'artifacts' / 'nnt1_stale.nnt').write_bytes(b'an earlier step')   # history a release must not carry
    out = P.release(argparse.Namespace(dir=str(directory)))
    result = json.loads((build / 'result.json').read_text())
    assert (out / 'result.json').read_bytes() == (build / 'result.json').read_bytes()   # same served policy id
    assert sorted(p.stem for p in (out / 'artifacts').iterdir()) == sorted(i['artifact'] for i in result['artifacts'])
    served = load_native_engine(str(tmp_path / 'native.pt'), device=engine.device)
    assert [e['id'] for e in bind_adapter_release(served, out)] == [i['artifact'] for i in result['artifacts']]


def test_fit_engines_do_not_keep_the_checkpoint_training_state(tmp_path):
    # the training state (optimizer moments) held 3.3 GB of host memory through an LFM3 program fit (2026-11-20)
    from natlang_neuralese.serve import load_native_engine
    engine_fixture(tmp_path, 'lfm', service_template=True)
    kept = load_native_engine(str(tmp_path / 'native.pt'), device='cpu')
    dropped = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', training_state=False)
    assert dropped.native_training_state == {}
    assert kept.native_identity == dropped.native_identity


def test_plan_keeps_symlinked_paths_as_written(tmp_path):
    """2026-11-20: plans resolved rows.jsonl through an archive symlink; when the directory moved back, every fit of
    the chain failed on the vanished archive path. The plan binds to the path as written."""
    real, link = tmp_path / 'archive' / 'prog', tmp_path / 'data' / 'prog'
    real.mkdir(parents=True)
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)
    (real / 'split.json').write_text(json.dumps({'family': 'fam'}))
    (real / 'rows.jsonl').write_text('{}\n')
    ckpt = tmp_path / 'ckpt-link.pt'
    ckpt.symlink_to(real / 'rows.jsonl')
    P.main(['plan', '--dir', str(link), '--checkpoint', str(ckpt), '--checkpoint-sha256', 'a' * 64, '--base', 'b'])
    body = json.loads((real / 'fit' / 'plan.json').read_text())
    assert body['rows'] == str(link / 'rows.jsonl')
    assert body['engine']['checkpoint'] == body['student']['checkpoint'] == str(ckpt)


def test_colab_fit_runs_as_a_preemptible_tenant_and_brings_the_release_home(tmp_path, monkeypatch):
    """DECISIONS 2026-11-24 "program tuning on the A100": the fit's paths move to the VM, it runs under a preemptible claim beside the
    actor at the tenant commit, an eviction returns non-zero (rerun resumes), and only a whole release lands in DIR."""
    import pop_rl_actor as A
    prog = tmp_path / 'prog'
    (prog / 'fit').mkdir(parents=True)
    (prog / 'split.json').write_text(json.dumps({'family': 'ext_fam'}))
    (prog / 'rows.jsonl').write_text('{"r": 1}\n')
    P.main(['plan', '--dir', str(prog), '--checkpoint', str(prog / 'rows.jsonl'), '--checkpoint-sha256', 'c' * 64,
            '--base', '/hf', '--steps', '3'])
    calls, plans, rcs = [], {}, [75, 0]
    monkeypatch.setattr(A, 'colab_tenant_checkout', lambda h, cfg, c: {**cfg, 'repo': '/t', 'admission': '/t/adm.py', 'commit': c})
    monkeypatch.setattr(A, 'ensure_remote_checkpoint', lambda *a: calls.append(('ckpt', a[3])) or 'present')
    monkeypatch.setattr(A, 'ship_file', lambda local, remote, *a: calls.append(('ship', remote)) or 'drive')
    monkeypatch.setattr(A, 'colab_hardware', lambda h: {'gpu_name': 'NVIDIA A100-SXM4-80GB', 'vram_total_gib': 80.0,
                                                        'host_ram_total_gib': 167.0})
    monkeypatch.setattr(A, 'log', lambda **e: None)

    def ssh_host(host, command, *, input=None, timeout=600, check=True):
        if input is not None and 'plan.json' in command:
            plans['plan'] = json.loads(input)
            assert f'{json.loads(input)["engine"]["store_scratch"]}' in command   # created before the fit runs
        if command.startswith('ls '):
            return 'nnt1_abc.nnt\n'
        return ''
    monkeypatch.setattr(A, 'ssh_host', ssh_host)
    monkeypatch.setattr(A, 'run_detached', lambda host, cfg, remote, command: calls.append(('run', command)) or rcs.pop(0))

    def pull(remote, local, host, attempts=5):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text(json.dumps({'artifacts': []}) if remote.endswith('.json') else 'x')
        return 'sha'
    monkeypatch.setattr(A, 'pull_verified', pull)
    args = ['colab-fit', '--dir', str(prog), '--host', 'colab-g4', '--commit', 'f' * 40]
    assert P.main(args) == 75 and not (prog / 'fit' / 'release').exists()      # evicted: nothing lands
    assert (prog / 'fit' / 'colab-claim.json').exists()                         # the DGX chain skips it
    assert P.main(args) == 0 and (prog / 'fit' / 'release' / 'artifacts' / 'nnt1_abc.nnt').exists()
    body = plans['plan']
    remote = '/content/natlang-work/runs/tenant-fit-ext_fam'
    assert body['rows'] == f'{remote}/rows.jsonl' and body['engine']['checkpoint'].endswith('/checkpoints/' + 'c' * 64 + '.pt')
    assert body['engine']['base'].endswith('/base-lfm3') and body['recipe']['steps'] == 3
    command = [c for c in calls if c[0] == 'run'][-1][1]
    assert command[1] == '/t/adm.py' and '--preemptible' in command and command[command.index('--beside-label') + 1] == 'actor-'
    # outside MPS (a tenant's GPU fault must not end the actor's context): no SM share, a pipe with no daemon
    assert not any(c.startswith('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=') for c in command)
    assert any(c.startswith('CUDA_MPS_PIPE_DIRECTORY=') and c.endswith('/mps/tenant-no-mps') for c in command)
    assert command[-4:] == ['--plan', f'{remote}/fit/plan.json',
                                                                                  '--out', f'{remote}/fit/build']
    assert json.loads((prog / 'fit' / 'colab.json').read_text())['tenant_commit'] == 'f' * 40
    assert P.main(args) == 0                                                    # present: nothing runs again
    (prog / 'fit' / 'release' / 'result.json').unlink()
    try:
        P.main(args)
        raise AssertionError('a local fit state must not be taken over silently')
    except SystemExit as stop:
        assert 'local fit' in str(stop)


def test_a_subset_is_a_stable_pinned_manifest_of_the_split(tmp_path):
    from natlang_neuralese.rl.held_eval import write_manifest, MANIFEST_SCHEMA
    prog = tmp_path / 'prog'
    rows = [{'id': f't{i}', 'family': 'fam'} for i in range(10)]
    write_manifest(prog / 'dev', {'schema': MANIFEST_SCHEMA, 'program': {'family': 'fam', 'split': 'dev'},
                                  'tasks': [{'id': r['id']} for r in rows], 'excluded_split_groups': ['g']}, rows)
    (prog / 'dev' / 'engine.json').write_text('{"e": 1}')
    P.main(['subset', '--dir', str(prog), '--tasks', '4'])
    sub = json.loads((prog / 'dev4' / 'manifest.json').read_text())
    cases = [json.loads(line) for line in (prog / 'dev4' / 'cases.jsonl').read_text().splitlines()]
    assert len(sub['tasks']) == len(cases) == 4 and {t['id'] for t in sub['tasks']} == {c['id'] for c in cases}
    assert sub['subset_of']['tasks'] == 4 and (prog / 'dev4' / 'engine.json').read_text() == '{"e": 1}'
    P.main(['subset', '--dir', str(prog), '--tasks', '6', '--name', 'dev6'])
    six = {t['id'] for t in json.loads((prog / 'dev6' / 'manifest.json').read_text())['tasks']}
    assert {t['id'] for t in sub['tasks']} < six                                # stable order: a larger subset extends it


def test_colab_bank_ships_the_fits_runs_as_a_tenant_and_brings_the_held_releases_home(tmp_path, monkeypatch):
    import pop_rl_actor as A
    dirs = []
    for name in ('ext_a', 'ext_b', 'ext_held'):
        prog = tmp_path / name
        (prog / 'fit' / 'release' / 'artifacts').mkdir(parents=True)
        (prog / 'split.json').write_text(json.dumps({'family': name}))
        (prog / 'rows.jsonl').write_text('{"r": 1}\n')
        (prog / 'fit' / 'plan.json').write_text(json.dumps({'manifest': {'base': {'checkpoint_sha256': 'c' * 64}},
                                                            'engine': {'checkpoint': '/ck.pt'}}))
        dirs.append(prog)
    calls = []
    monkeypatch.setattr(A, 'colab_tenant_checkout', lambda h, cfg, c: {**cfg, 'repo': '/t', 'admission': '/t/adm.py', 'commit': c})
    monkeypatch.setattr(A, 'ensure_remote_checkpoint', lambda *a: 'present')
    monkeypatch.setattr(A, 'ship_file', lambda local, remote, *a: calls.append(('ship', remote)))
    monkeypatch.setattr(A, 'ship_release', lambda h, cfg, local, remote: calls.append(('release', remote)) or remote)
    monkeypatch.setattr(A, 'copy_resumable', lambda *a, **k: None)
    monkeypatch.setattr(A, 'log', lambda **e: None)
    monkeypatch.setattr(A, 'ssh_host', lambda host, command, **k: 'nnt1_x.nnt\n' if command.startswith('ls ') else '')
    monkeypatch.setattr(A, 'run_detached', lambda host, cfg, remote, command: calls.append(('run', command)) or 0)

    def pull(remote, local, host, attempts=5):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text('{}')
    monkeypatch.setattr(A, 'pull_verified', pull)
    out = tmp_path / 'bank-1'
    assert P.main(['colab-bank', '--programs', str(dirs[0]), str(dirs[1]), '--held', str(dirs[2]), '--out', str(out),
                   '--host', 'colab-g4', '--commit', 'f' * 40, '--steps', '2']) == 0
    command = [c for c in calls if c[0] == 'run'][0][1]
    assert '--preemptible' in command and command[command.index('--vram-gib') + 1] == '14.0'
    remote = '/content/natlang-work/runs/tenant-bank-bank-1'
    bank = command[command.index('bank'):]
    assert bank[bank.index('--held') + 1] == f'{remote}/programs/ext_held' and '--steps' in bank
    assert [c[1] for c in calls if c[0] == 'release'] == [f'{remote}/programs/ext_a/fit/release',
                                                          f'{remote}/programs/ext_b/fit/release']
    assert (out / 'adapt' / 'ext_held' / 'artifacts' / 'nnt1_x.nnt').exists() and (out / 'bank-manifest.json').exists()


def test_two_colab_hosts_get_their_own_forwards_and_a_dedicated_session_needs_no_actor(monkeypatch):
    import pop_rl_actor as A
    assert A.colab_forward_offset('colab-g4') == 10161 and A.colab_forward_offset('colab-g4b') == 11161
    assert A.colab_forward_offset('colab-x') not in (10161, 11161)
    monkeypatch.setenv('NATLANG_COLAB_FORWARD_OFFSETS', 'colab-x=13161')
    assert A.colab_forward_offset('colab-x') == 13161
    cfg = A.host_config('colab', host='colab-g4b')
    tcfg = {**cfg, 'repo': '/t', 'admission': '/t/adm.py'}
    alone = A.colab_tenant_command(cfg, tcfg, {'vram_gib': 12, 'host_ram_gib': 16}, 'fit', ['x'], beside=None)
    beside = A.colab_tenant_command(cfg, tcfg, {'vram_gib': 12, 'host_ram_gib': 16}, 'fit', ['x'])
    assert '--beside-label' not in alone and '--preemptible' in alone
    assert beside[beside.index('--beside-label') + 1] == 'actor-'


def test_colab_fit_refuses_a_program_another_session_claimed(tmp_path):
    prog = tmp_path / 'prog'
    (prog / 'fit').mkdir(parents=True)
    (prog / 'split.json').write_text(json.dumps({'family': 'ext_fam'}))
    (prog / 'fit' / 'plan.json').write_text('{}')
    (prog / 'fit' / 'colab-claim.json').write_text(json.dumps({'host': 'colab-g4'}))
    try:
        P.main(['colab-fit', '--dir', str(prog), '--host', 'colab-g4b', '--commit', 'f' * 40])
        raise AssertionError('a program claimed by another session must not be fitted twice')
    except SystemExit as stop:
        assert 'claimed by colab-g4' in str(stop)


def test_a_merged_release_equals_its_summed_free_lora_stack():
    """`merge`: a stack of free-LoRA artifacts becomes one factor pair per backbone target whose product is the sum of
    each artifact's scale x alpha / rank x B @ A, and the learner's merge applies it to the base weight."""
    import types
    import torch
    from natlang_neuralese.train.lfm3_learner import merge_policy_state
    g = torch.Generator().manual_seed(0)
    name = 'port.hf.model.layers.0.conv.in_proj.weight'

    def artifact(rank, alpha):
        B, A = torch.randn(6, rank, generator=g), torch.randn(rank, 4, generator=g)
        record = types.SimpleNamespace(kind='adapter-coefficients', tensors={'lora_B.' + name: B, 'lora_A.' + name: A},
                                       provenance={'adapter_spec': {'kind': 'free-lora', 'alpha': alpha,
                                                                    'parameter_names': [name]}})
        return record, (alpha / rank) * B @ A
    (one, d1), (two, d2) = artifact(2, 4.0), artifact(3, 3.0)
    tensors, scales, specs = P.merged_factors({'a': one, 'b': two}, [('a', 1.0), ('b', 0.5)])
    key = 'model.layers.0.conv.in_proj.weight'
    assert set(scales) == {key} and len(specs) == 2
    assert torch.allclose(tensors[f'lora_B.{key}'] @ tensors[f'lora_A.{key}'], d1 + 0.5 * d2, atol=1e-5)
    base = torch.randn(6, 4, generator=g)
    state = {'backbone_state': {key: base, 'model.embed_tokens.weight': torch.zeros(2, 4)}}
    backbone, _, _ = merge_policy_state(state, tensors, scales, dtype=torch.float32)
    assert torch.allclose(backbone[key], base + d1 + 0.5 * d2, atol=1e-5)
    with pytest.raises(SystemExit):
        P.merged_factors({'a': one}, [('missing', 1.0)])
