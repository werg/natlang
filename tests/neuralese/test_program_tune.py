"""Per-program LoRA tuning driver (scripts/neuralese_data/program_tune.py): split, verified turn rows, the plan, and that
its rows lower and fit through the tuner on a tiny native engine."""
import json
import sys
from pathlib import Path

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
                              target_pattern=r'self_attn\.(q_proj|k_proj|v_proj|out_proj)|conv\.(in_proj|out_proj)')
    P.bank(args)
    bank = tmp_path / 'bank'
    pool = json.loads((bank / 'pool' / 'pool.json').read_text())
    assert {name: f['role'] for name, f in pool['families'].items()} == {'fam0': 'meta-train', 'fam1': 'meta-train',
                                                                         'fam2': 'meta-train', 'held': 'held'}
    assert json.loads((bank / 'learner' / 'state.json').read_text())['step'] == 1
    assert (bank / 'bank-manifest.json').exists()
    served = load_native_engine(str(tmp_path / 'native.pt'), device=engine.device)
    entries = bind_adapter_release(served, bank / 'adapt' / 'held')
    assert served.rl_policy_binding['adapters']['stack'] == entries
