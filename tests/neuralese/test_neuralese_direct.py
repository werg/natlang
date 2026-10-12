"""Neuralese direct v1 (plans/neuralese/NEURALESE_DIRECT_V1.md): the two projections, their step-0 identities,
causality and segment resets, and the recipe wiring."""
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / 'training/neuralese/recipes'


def _trained(module):
    """Give every zero-initialised output projection nonzero weights (a trained stack)."""
    torch.manual_seed(1)
    with torch.no_grad():
        for block in module.stack.blocks:
            block.down.weight.normal_(0, 0.1)
            block.taps.normal_(0, 0.5)
    return module


def test_token_choice_projection_is_the_identity_and_logits_equal_base_at_initialization():
    from natlang_neuralese.model.neuralese_projection import TokenChoiceProjection
    torch.manual_seed(0)
    hidden = torch.randn(2, 9, 16)
    head = torch.nn.Linear(16, 50, bias=False)
    norm = torch.nn.RMSNorm(16)
    projection = TokenChoiceProjection(16)
    assert torch.equal(projection(hidden), hidden)
    assert torch.equal(head(norm(projection(hidden))), head(norm(hidden)))


def test_neuralese_projection_is_the_soft_token_mixture_at_initialization():
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection, soft_token_mixture
    torch.manual_seed(0)
    hidden = torch.randn(2, 9, 16)
    table = torch.randn(50, 16)
    head = torch.nn.Linear(16, 50, bias=False)
    mixture = soft_token_mixture(head, table, hidden, chunk=4)
    assert torch.allclose(mixture, torch.softmax(head(hidden), -1) @ table, atol=1e-6)
    projection = NeuraleseProjection(16, hidden=8)
    assert torch.equal(projection(hidden, mixture), mixture)
    # Trainable: a gradient reaches the correction's output projection from the first step.
    projection(hidden, mixture).square().sum().backward()
    assert projection.stack.blocks[-1].down.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['neuralese', 'token_choice'])
def test_projections_are_causal_and_reset_at_segment_boundaries(kind):
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection, TokenChoiceProjection
    torch.manual_seed(0)
    module = _trained(NeuraleseProjection(8, hidden=4) if kind == 'neuralese' else TokenChoiceProjection(8, hidden=4))
    hidden = torch.randn(1, 10, 8)
    mixture = torch.zeros(1, 10, 8)
    segments = torch.tensor([[-1, -1, 0, 0, 0, 1, 1, 1, 1, 1]])

    def run(h):
        return module(h, mixture, segments) if kind == 'neuralese' else module(h, segments)
    base = run(hidden)
    later = hidden.clone(); later[:, 7] += 1.0  # a later position never changes an earlier output
    assert torch.equal(run(later)[:, :7], base[:, :7])
    other = hidden.clone(); other[:, 2:5] += 1.0  # segment 0 never leaks into segment 1
    assert torch.equal(run(other)[:, 5:], base[:, 5:])
    context = hidden.clone(); context[:, :2] += 1.0  # context positions (no segment) never feed a segment
    assert torch.equal(run(context)[:, 2:], base[:, 2:])


def test_segment_ids_are_turns_inside_the_supervised_span():
    from natlang_neuralese.train.direct_sft import segment_ids
    roles = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 1]])
    segments = segment_ids(roles, 2, 8, 1, 'cpu')
    assert segments.tolist() == [[-1, -1, 0, 0, 1, 1, 1, 2]]
    assert segment_ids(None, 3, 6, 2, 'cpu').tolist() == [[-1, -1, -1, 0, 0, 0]] * 2


def test_direct_recipes_resolve_for_both_lines_and_gate_on_qualification():
    from natlang_neuralese.train.recipe import (HANDLERS, require_gate, resolve_recipe_data, stage_parameter_args,
                                                stage_quantization)
    assert HANDLERS['neuralese_direct']['module'] == 'natlang_neuralese.train.direct_sft'
    mellum = resolve_recipe_data(RECIPES / 'neuralese-direct-mellum-v1.json')
    lfm = resolve_recipe_data(RECIPES / 'neuralese-direct-lfm-v1.json')
    for recipe in (mellum, lfm):
        stage = recipe['stages'][0]
        assert stage['kind'] == 'neuralese_direct'
        defaults = recipe['stage_parameter_defaults']['neuralese_direct']
        assert defaults['token_choice_projection'] is False and defaults['tokens'] > defaults['target_tokens']
        args = stage_parameter_args({**defaults, **stage['parameters']})
        assert '--no-token-choice-projection' in args
        assert defaults['value_margin'] > 0 and '--value-margin' in args  # report-only (owner 2026-10-11)
        assert 0 < defaults['drift_every'] < defaults['eval_every'] and '--drift-every' in args  # the guard sees the run
        assert defaults['drift_gate_every'] >= defaults['drift_every'] and 0 < defaults['drift_gate_rise'] < defaults['drift_stop']
        assert defaults['gate_response_kl'] == defaults['drift_stop'] and defaults['drift_ce_windows'] > 0
    assert mellum['stage_parameter_defaults']['neuralese_direct']['moe_kernel'] == 'grouped'
    assert mellum['quantization']['preserve']['weight'] == 0.0
    stage = mellum['stages'][0]
    component = stage_quantization(mellum, stage, {**mellum['stage_parameter_defaults']['neuralese_direct'],
                                                   **stage['parameters']}, [])
    assert component['stage'] == {'id': 'direct_sft', 'progress': [0.0, 0.35], 'start': 0,
                                  'end': stage['parameters']['steps']}
    assert 'quantization' not in lfm
    with pytest.raises(ValueError, match='not qualified'):
        require_gate({'qualified': False}, 'neuralese_direct')
    require_gate({'qualified': True}, 'neuralese_direct')


def test_failed_assistant_turns_are_masked_from_every_loss():
    """An assistant turn answered by a runtime failure (rejected action, discarded eval, refused result) is context
    only: zero weight in both halves of the whole-trajectory supervision."""
    from natlang_neuralese.train.text_supervision import (ROLE_CODES, balanced_position_weights,
                                                          failed_assistant_mask)
    A, T, O, U = (ROLE_CODES.index(n) for n in ('assistant_reply', 'tool', 'other', 'user'))
    # user | assistant(1) | tool: rejected | assistant(2) | tool: ok | assistant(3, last turn)
    labels = [U, U, A, A, O, T, T, A, A, O, T, A]
    ids = list(range(len(labels)))
    texts = {(5, 6): '<tool_response>\nrejected\ncode: bad-action', (10,): '<tool_response>\nok'}

    def decode(tokens):
        return texts[tuple(tokens)]
    mask = failed_assistant_mask(ids, labels, decode)
    assert list(mask) == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    for body in ('1:1 invalid-binding: x\nNothing else from this eval was kept.',
                 'false\nThis is not a valid { a: string }, so it is not the result: return: type-mismatch'):
        texts[(5, 6)] = '<tool_response>\n' + body
        assert list(failed_assistant_mask(ids, labels, decode))[2:4] == [1, 1]
    span = torch.tensor([ids])
    exclude = torch.tensor([list(failed_assistant_mask(ids, labels, decode))]).bool()
    weights = balanced_position_weights(span, [8], roles=torch.tensor([labels]), exclude=exclude)
    assert (weights[0, 2:4] == 0).all() and (weights[0, 7:] > 0).all()
    assert torch.allclose(weights.sum(), torch.tensor(float(len(labels))))
    everything = balanced_position_weights(span, [None], roles=torch.tensor([labels]),
                                           exclude=torch.ones_like(exclude))
    assert (everything == 0).all()
    unchanged = balanced_position_weights(span, [8], roles=torch.tensor([labels]))
    assert torch.equal(unchanged, balanced_position_weights(span, [8], roles=torch.tensor([labels]),
                                                            exclude=torch.zeros_like(exclude)))


def test_monitor_reference_is_captured_from_the_model_itself():
    """The drift reference of a lineage without a teacher artifact: the model's own greedy answers and top-k, so the
    held KL of the unchanged model against it is zero."""
    import types
    from natlang_neuralese.maple.generation_gate import capture_reference, held_kl
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(40, 8)
            self.lm_head = torch.nn.Linear(8, 40, bias=False)
            self.model = lambda input_ids: types.SimpleNamespace(last_hidden_state=self.embed(input_ids))

        def generate(self, ids, max_new_tokens, **_):
            out = ids
            for _ in range(max_new_tokens):
                nxt = self.lm_head(self.embed(out[:, -1:])).argmax(-1)
                out = torch.cat([out, nxt], 1)
            return out
    model = Tiny()
    render = lambda tokenizer, prompt: prompt['ids']
    reference = capture_reference(model, None, [{'ids': [1, 2, 3]}, {'ids': [4, 5]}], render, eos=39, max_new=4, k=40)
    assert [len(x) for x in reference['ids']] == [7, 6] and reference['prompt_len'] == [3, 2]
    report = held_kl(model, reference, rows=2)
    assert report['held_kl'] < 1e-3 and report['held_response_agreement'] == 1.0


def test_declared_json_stage_components_serialize_as_json():
    from natlang_neuralese.train.recipe import stage_parameter_args
    args = stage_parameter_args({'monitor': {'data': {'path': 'x.jsonl', 'sha256': 'a' * 64}}})
    assert args[0] == '--monitor' and json.loads(args[1]) == {'data': {'path': 'x.jsonl', 'sha256': 'a' * 64}}


def test_declared_context_only_spans_are_masked_with_the_failed_turns():
    """A producer's ``context_only_spans`` (excluded bad assistant targets kept in a retained history) are context
    only, in the row's token_ids coordinates, on top of the runtime-failure mask; malformed spans are refused."""
    from natlang_neuralese.train.direct_sft import context_only_mask, context_only_spans
    from natlang_neuralese.train.text_supervision import ROLE_CODES
    A, O, U = (ROLE_CODES.index(n) for n in ('assistant_reply', 'other', 'user'))
    row = {'token_ids': list(range(10)), 'context_only_spans': [[2, 4], [7, 9]]}
    spans = context_only_spans(row)
    source = [100] + row['token_ids'] + [101]  # framed: open token, token_ids, close token
    labels = [O] + [U, U, A, A, A, U, U, A, A, A] + [O]
    mask = context_only_mask(source, labels, spans, lambda tokens: '')
    assert list(mask) == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    assert list(context_only_mask(source, None, spans, None)) == list(mask)
    for bad in ([[4, 2]], [[2, 5], [4, 6]], [[0, 11]], [[1.0, 2]], [[1]]):
        with pytest.raises(ValueError, match='context_only_spans'):
            context_only_spans({**row, 'context_only_spans': bad})
    with pytest.raises(ValueError, match='native token_ids'):
        context_only_spans({'context_only_spans': [[0, 1]]})


def test_a_direct_lineage_loads_without_warm_up_stages_and_old_lineages_keep_the_rule(tmp_path):
    """Owner decision 2026-10-11: a recipe of neuralese_direct stages needs no token_identity /
    causal_embedding_distillation stages; its declared step-0 identity check is the identity qualification."""
    from natlang_neuralese.train.recipe import DIRECT_IDENTITY_QUALIFICATION, load_recipe, resolve_recipe_data
    for name in ('neuralese-direct-mellum-v1.json', 'neuralese-direct-lfm-v1.json'):
        assert load_recipe(RECIPES / name)['identity_qualification'] == DIRECT_IDENTITY_QUALIFICATION
    recipe = resolve_recipe_data(RECIPES / 'neuralese-direct-v1.json')
    undeclared = {k: v for k, v in recipe.items() if k != 'identity_qualification'}
    (tmp_path / 'undeclared.json').write_text(json.dumps(undeclared))
    with pytest.raises(ValueError, match='identity_qualification'):
        load_recipe(tmp_path / 'undeclared.json')
    old = resolve_recipe_data(RECIPES / 'raw-recurrence-mellum-v12.json')
    old['stages'] = [s for s in old['stages'] if s['kind'] not in ('token_identity', 'causal_embedding_distillation')]
    (tmp_path / 'old.json').write_text(json.dumps(old))
    with pytest.raises(ValueError):
        load_recipe(tmp_path / 'old.json')


def test_raw_token_mellum_anchor_opens_on_the_direct_serving_path(tmp_path, monkeypatch):
    """direct-mellum-v2's clean anchor: maple.foundation_heads --profile raw-token-v1 at the full stack, which
    serve.load_engine restores without read markers (the direct writer's precondition)."""
    test_mellum_port = pytest.importorskip('test_mellum_port')
    from natlang_neuralese.maple import foundation_heads, student
    from natlang_neuralese.maple.model import load_maple
    from natlang_neuralese.serve import load_engine
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    test_mellum_port.tiny_mellum(model_dir)

    class Vocab:  # the tiny model has no tokenizer files: Mellum's spare added tokens are the markers
        def get_vocab(self):
            return {'<|extra_token_7|>': 33, '<|extra_token_8|>': 34}

        eos_token_id = 2
        # Already bound to Mellum's declared history policy (the stub has no chat template to probe).
        from natlang_neuralese.serve.chat import declared_history_reasoning
        natlang_history_reasoning = {'policy': declared_history_reasoning('mellum')}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)

    def tiny_student(model, state=None, device='cpu', **_):
        loaded = load_maple(model, device='cpu', dtype=torch.float32, ternary_attention=False)
        loaded.config.model_type = 'mellum'
        return loaded, Vocab()
    monkeypatch.setattr(foundation_heads, 'load_student', tiny_student)
    monkeypatch.setattr(student, 'load_student', tiny_student)
    layers = json.loads((model_dir / 'config.json').read_text())['num_hidden_layers']
    out = tmp_path / 'heads.pt'
    foundation_heads.main(['--model', str(model_dir), '--precision', 'bf16', '--profile', 'raw-token-v1',
                           '--cutoff', str(layers), '--stop-source', 'final', '--out', str(out)])
    saved = torch.load(out, weights_only=False)
    assert saved['port_config']['profile'] == 'raw-token-v1' and saved['port_config']['cutoff'] == layers
    engine = load_engine(heads_checkpoint=str(out), device='cpu')
    assert not engine.heads.read_markers and engine.heads.profile == 'raw-token-v1'
    assert engine.heads.cutoff == engine.backbone.num_layers
    with pytest.raises(SystemExit):  # the legacy profile keeps its inside-the-stack cutoff
        foundation_heads.main(['--model', str(model_dir), '--precision', 'bf16', '--cutoff', str(layers),
                               '--out', str(tmp_path / 'legacy.pt')])


def test_direct_sft_resume_copies_through_the_shared_restore_and_keeps_no_view_of_the_state():
    """The v2 resume leak (39 GB host anon): direct_sft restores backbone values through text_warmup's
    copy_restored_values, so no loop variable in main() keeps the mmap'd state alive."""
    import ast
    import inspect
    from natlang_neuralese.train import direct_sft
    tree = ast.parse(inspect.getsource(direct_sft.main))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and "student_parameters" in ast.unparse(node.iter)]
    assert not loops, "restore student_parameters with copy_restored_values, never a loop in main()"
    assert "copy_restored_values" in inspect.getsource(direct_sft.main)


def test_direct_monitor_and_gate_fields_change_at_a_handoff():
    """The drift monitor's and the gate's fields are operational: a lineage resumes across their addition, change or
    retirement (v2: drift_gate_now -> drift_gate_rise, drift_ce_windows, gate_response_kl)."""
    import copy
    from natlang_neuralese.train.recipe import _without_operational, resolve_recipe_data
    new = resolve_recipe_data(RECIPES / 'neuralese-direct-mellum-v2.json')
    old = copy.deepcopy(new)
    defaults = old['stage_parameter_defaults']['neuralese_direct']
    for key in ('drift_gate_rise', 'drift_ce_windows', 'gate_response_kl', 'drift_gate_every', 'drift_every'):
        defaults.pop(key, None)
    defaults['drift_gate_now'] = 0.35
    assert _without_operational(old) == _without_operational(new)
    changed = copy.deepcopy(new)
    changed['stage_parameter_defaults']['neuralese_direct']['latent_lr'] = 1e-5
    assert _without_operational(changed) != _without_operational(new)  # training semantics still refuse


def _windows(documents, per_document=2):
    return [{'document': f'd{d}', 'ids': list(range(10 + (d * 7 + w * 13) % 40)), 'prefix': 2, 'cohort': 'native', 'w': w}
            for d in range(documents) for w in range(per_document)]


def test_epoch_schedule_sees_every_window_once_per_epoch_without_replacement():
    from natlang_neuralese.train.text_supervision import CohortDocumentSampler, epoch_schedule
    windows = _windows(50)
    sampler = CohortDocumentSampler(windows, None, cohort_key=lambda w: w['cohort'], document_key=lambda w: w['document'])
    length = lambda w: len(w['ids']) - 1
    plan = epoch_schedule(sampler, seed=3, length=length, epochs=2.0, max_tokens=160)
    seen = [id(w) for batch, _ in plan for w in batch]
    assert len(seen) == 200 and len(set(seen)) == 100 and abs(plan[-1][1] - 2.0) < 1e-9
    first = [id(w) for batch, covered in plan if covered <= 1.0 for w in batch]
    assert sorted(first) == sorted(id(w) for w in windows)  # epoch 1 is a permutation of every window
    assert all(len(batch) * max(map(length, batch)) <= 160 or len(batch) == 1 for batch, _ in plan)
    assert [[id(w) for w in b] for b, _ in plan] == [[id(w) for w in b] for b, _ in
                                                     epoch_schedule(sampler, seed=3, length=length, epochs=2.0,
                                                                    max_tokens=160)]  # resumable: a pure function
    half = epoch_schedule(sampler, seed=3, length=length, epochs=0.5, max_tokens=0, max_items=1)
    assert len(half) == 50 and abs(half[-1][1] - 0.5) < 1e-9
    legacy = epoch_schedule(sampler, seed=3, length=length, updates=250)
    assert len(legacy) == 250 and all(len(b) == 1 for b, _ in legacy)
    with pytest.raises(ValueError):
        epoch_schedule(sampler, seed=3, length=length, epochs=1.0, updates=10)


def test_sampling_groups_balance_candidate_documents_and_resume_deterministically():
    from natlang_neuralese.train.text_supervision import CohortDocumentSampler, epoch_schedule
    windows = [
        {'document': 'item-a-choice-1', 'sampling_group': 'source-item:a', 'cohort': 'native', 'id': 'a1-w1', 'ids': [1, 2]},
        {'document': 'item-a-choice-1', 'sampling_group': 'source-item:a', 'cohort': 'native', 'id': 'a1-w2', 'ids': [1, 2, 3]},
        {'document': 'item-a-choice-2', 'sampling_group': 'source-item:a', 'cohort': 'native', 'id': 'a2-w1', 'ids': [1, 2, 3, 4]},
        {'document': 'item-b-choice-1', 'sampling_group': 'source-item:b', 'cohort': 'native', 'id': 'b1-w1', 'ids': [1, 2]},
    ]
    sampler = CohortDocumentSampler(windows, None, cohort_key=lambda w: w['cohort'],
                                    document_key=lambda w: w['document'],
                                    sampling_group_key=lambda w: w.get('sampling_group'))
    receipt = sampler.receipt()
    assert receipt['policy'] == 'cohort-then-sampling-group-then-document-then-item/1'
    assert receipt['sampling_groups'] == {'native': 2}
    assert receipt['candidate_documents'] == {'native': 3}
    assert receipt['draws_per_group_per_epoch'] == 1
    assert receipt['group_draw_weights_equal'] is True
    assert receipt['candidate_coverage_epochs'] == {'native': 2}

    def group_draws(epoch):
        docs = sampler.epoch(epoch, seed=23)
        assert len(docs) == 2  # one candidate document per source item
        return {doc[0]['sampling_group']: doc[0]['document'] for doc in docs}

    first, second = group_draws(0), group_draws(1)
    assert set(first) == {'source-item:a', 'source-item:b'}
    assert set(second) == set(first)
    assert first['source-item:a'] != second['source-item:a']  # candidate actions rotate across epochs
    assert first == group_draws(0)  # epoch reconstruction is deterministic

    length = lambda w: len(w['ids'])
    partial = epoch_schedule(sampler, seed=23, length=length, updates=3)
    resumed = epoch_schedule(sampler, seed=23, length=length, updates=6)
    assert [[w['id'] for w in batch] for batch, _ in partial] == \
        [[w['id'] for w in batch] for batch, _ in resumed[:3]]

    # With one cohort, the draw process chooses the source item before its
    # candidate document: extra candidate actions do not multiply item weight.
    import random
    counts = {'source-item:a': 0, 'source-item:b': 0}
    rng = random.Random(20261012)
    for _ in range(20000):
        counts[sampler.sample(rng)['sampling_group']] += 1
    assert abs(counts['source-item:a'] - counts['source-item:b']) < 500

    # Untagged corpora preserve the pre-grouped sampler's seeded RNG sequence.
    legacy_windows = [
        {'document': 'doc-a', 'cohort': 'native', 'id': 'a1'},
        {'document': 'doc-a', 'cohort': 'native', 'id': 'a2'},
        {'document': 'doc-b', 'cohort': 'native', 'id': 'b1'},
    ]
    legacy_sampler = CohortDocumentSampler(legacy_windows, None, cohort_key=lambda w: w['cohort'],
        document_key=lambda w: w['document'])
    actual_rng, previous_rng = random.Random(91), random.Random(91)
    actual = [legacy_sampler.sample(actual_rng)['id'] for _ in range(20)]
    previous = []
    for _ in range(20):
        previous_rng.choices(['native'], weights=[1.], k=1)
        previous.append(previous_rng.choice(previous_rng.choice(legacy_sampler.documents['native']))['id'])
    assert actual == previous


def test_direct_phases_resolve_in_epochs_with_a_derived_quantization_end():
    from natlang_neuralese.train.recipe import resolve_recipe_data, stage_parameter_args, stage_quantization
    from natlang_neuralese.train.quantization import Quantization
    for name in ('neuralese-direct-mellum-v3.json', 'neuralese-direct-lfm-v2.json'):
        recipe = resolve_recipe_data(RECIPES / name)
        assert [s['id'] for s in recipe['stages']] == ['direct_warmup', 'direct_sft']
        assert recipe['stages'][1]['requires'] == ['direct_warmup']
        for stage, epochs in zip(recipe['stages'], (0.5, 2.0)):
            parameters = {**recipe['stage_parameter_defaults']['neuralese_direct'], **stage['parameters']}
            assert parameters['epochs'] == epochs and 'steps' not in parameters and parameters['batch_tokens'] > 0
            assert '--epochs' in stage_parameter_args(parameters)
    mellum = resolve_recipe_data(RECIPES / 'neuralese-direct-mellum-v3.json')
    stage = mellum['stages'][1]
    component = stage_quantization(mellum, stage, {**mellum['stage_parameter_defaults']['neuralese_direct'],
                                                   **stage['parameters']}, [])
    assert component['stage']['end'] == 'derived'
    quant = Quantization(component)
    with pytest.raises(ValueError, match='not bound'):
        quant.progress(0)
    quant.stage['end'] = 346
    assert quant.progress(346) == pytest.approx(0.35)
    assert stage_quantization(mellum, mellum['stages'][0], {'epochs': 0.5}, []) is None  # the warm-up is BF16


def test_a_continued_phase_archives_its_predecessor_state_off_the_nvme(tmp_path, monkeypatch):
    import importlib.util
    import os
    from natlang_neuralese.train import direct_sft
    spec = importlib.util.spec_from_file_location('archive_to_external', ROOT / 'scripts' / 'archive_to_external.py')
    mover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mover)
    monkeypatch.setattr(mover, 'ROOTS', (tmp_path / 'runs',))
    monkeypatch.setattr(mover, 'ARCHIVE', tmp_path / 'archive')
    monkeypatch.setattr(importlib.util, 'module_from_spec', lambda _: mover)
    monkeypatch.setattr(spec.loader, 'exec_module', lambda _: None)
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', lambda *_: spec)
    stage = tmp_path / 'runs' / 'lineage' / 'direct_warmup'
    stage.mkdir(parents=True)
    slot = stage / 'checkpoint.pt'
    slot.write_bytes(os.urandom(1 << 16))
    os.link(slot, stage / 'best-checkpoint.pt')
    content = slot.read_bytes()
    record = direct_sft.archive_predecessor_state(slot)
    assert slot.is_symlink() and (stage / 'best-checkpoint.pt').is_symlink() and slot.read_bytes() == content
    assert json.loads((stage / 'checkpoint.pt.archived.json').read_text())['sha256'] == record['sha256']
    assert direct_sft.archive_predecessor_state(slot) is None  # already archived


def test_the_predecessor_archive_runs_beside_training_and_a_failure_keeps_the_state(tmp_path, monkeypatch, capsys):
    """Phase hand-off (owner 2026-10-12): the archive runs in a thread while the stage trains; the deferred disk
    reserve waits for it; an archive failure is logged and leaves the predecessor in place."""
    import inspect
    import threading
    from natlang_neuralese.train import direct_sft
    release, seen = threading.Event(), []
    monkeypatch.setattr(direct_sft, 'archive_predecessor_state', lambda slot: (release.wait(5), seen.append(slot)))
    archive = direct_sft.PredecessorArchive(tmp_path / 'checkpoint.pt')
    assert archive.running and not seen  # training is not waiting on the copy
    release.set()
    archive.wait()
    assert not archive.running and seen == [tmp_path / 'checkpoint.pt'] and archive.error is None

    def broken(slot):
        raise RuntimeError('verification failed')
    monkeypatch.setattr(direct_sft, 'archive_predecessor_state', broken)
    failed = direct_sft.PredecessorArchive(tmp_path / 'checkpoint.pt')
    failed.wait()
    assert 'verification failed' in failed.error and 'predecessor_state_archive_failed' in capsys.readouterr().out
    source = inspect.getsource(direct_sft.main)
    assert 'archive_predecessor_state(' not in source  # never synchronous on the critical path
    assert 'checkpoint_reserve_deferred_until_archive' in source and 'settle_reserve(wait=True)' in source


def test_a_continued_direct_phase_binds_both_sidecars_and_the_plan_check_requires_them(tmp_path, monkeypatch):
    """direct_sft refuses --continue-from without --continue-identity and --continue-monitor-reference (the v3p
    direct_sft failure, 2026-10-12): the runner binds all three from the predecessor phase, and load_recipe refuses a
    plan whose implicit continuation would lack one."""
    from natlang_neuralese.train import recipe as recipe_module
    from natlang_neuralese.train.recipe import (DIRECT_PHASE_CONTINUATION, direct_phase_continuation,
                                                direct_phase_predecessor, load_recipe, stage_input_args)
    for name in ('neuralese-direct-mellum-v3.json', 'neuralese-direct-lfm-v2.json'):
        recipe = load_recipe(RECIPES / name)
        first, second = recipe['stages']
        assert direct_phase_predecessor(recipe, first) is None
        assert direct_phase_predecessor(recipe, second) == first['id']
    args = stage_input_args(direct_phase_continuation(tmp_path / 'direct_warmup'), 'neuralese_direct')
    assert args == ['--continue-from', str(tmp_path / 'direct_warmup/checkpoint.pt'),
                    '--continue-identity', str(tmp_path / 'direct_warmup/identity-check.json'),
                    '--continue-monitor-reference', str(tmp_path / 'direct_warmup/monitor-reference.pt')]
    path = RECIPES / 'neuralese-direct-mellum-v3.json'
    for dropped in ('continue_identity', 'continue_monitor_reference'):
        monkeypatch.setattr(recipe_module, 'DIRECT_PHASE_CONTINUATION',
                            {k: v for k, v in DIRECT_PHASE_CONTINUATION.items() if k != dropped})
        with pytest.raises(ValueError, match='lacks input.*' + dropped):
            load_recipe(path)
    monkeypatch.setattr(recipe_module, 'DIRECT_PHASE_CONTINUATION', DIRECT_PHASE_CONTINUATION)
    declared = recipe_module.resolve_recipe_data(path)
    declared['stages'][1]['inputs']['continue_from'] = 'direct_text'
    (tmp_path / 'declared.json').write_text(json.dumps(declared))
    with pytest.raises(ValueError, match='continuation from its predecessor'):
        load_recipe(tmp_path / 'declared.json')


def test_union_r2_three_phase_recipes_and_the_mellum_resume_from_the_v3_warm_up():
    """neuralese-direct-v3 (2026-10-12): union r2, warm-up -> program header -> whole trajectory; Mellum's resume
    replaces its warm-up with the finished v3 warm-up as direct_sft's declared continuation inputs."""
    from natlang_neuralese.train.recipe import direct_phase_predecessor, load_recipe, stage_quantization
    for name in ('neuralese-direct-v3.json', 'neuralese-direct-mellum-v4.json', 'neuralese-direct-lfm-v3.json'):
        recipe = load_recipe(RECIPES / name)
        assert [s['id'] for s in recipe['stages']] == ['direct_warmup', 'direct_sft', 'direct_trajectory']
        assert direct_phase_predecessor(recipe, recipe['stages'][2]) == 'direct_sft'
        assert all('natlang-target-union-20261011-r2' in b['path'] for b in recipe['input_bindings'].values())
    resume = load_recipe(RECIPES / 'neuralese-direct-mellum-v4-from-v3-warmup.json')
    first, second = resume['stages']
    assert first['id'] == 'direct_sft' and first['requires'] == [] and direct_phase_predecessor(resume, first) is None
    assert {'continue_from', 'continue_identity', 'continue_monitor_reference'} <= set(first['inputs'])
    assert direct_phase_predecessor(resume, second) == 'direct_sft'
    component = stage_quantization(resume, second, {'epochs': 1.0}, [])
    assert component['stage']['progress'] == [0.75, 1.0]  # the shared curriculum (owner 2026-10-12)


def test_the_resume_identity_is_content_not_paths():
    """v4 2026-10-12: the predecessor moved from a staged NVMe copy to the HDD archive between launches; same bytes,
    same identity. A changed digest or a changed semantic option still refuses."""
    from natlang_neuralese.train.direct_sft import same_resume_identity
    saved = {'options': {'heads': '/a/heads.pt', 'text_data': '/staged/x/text.jsonl', 'latent_lr': 1e-5},
             'inputs': {'/a/heads.pt': 'h', '/staged/x/text.jsonl': 't', '/staged/c/checkpoint.pt': 'c'}}
    moved = {'options': {'heads': '/a/heads.pt', 'text_data': '/runs/text.jsonl', 'latent_lr': 1e-5},
             'inputs': {'/a/heads.pt': 'h', '/runs/text.jsonl': 't', '/hdd/archive/checkpoint.pt': 'c'}}
    assert same_resume_identity(saved, moved)
    assert not same_resume_identity(saved, {**moved, 'inputs': {**moved['inputs'], '/runs/text.jsonl': 'u'}})
    assert not same_resume_identity(saved, {**moved, 'options': {**moved['options'], 'latent_lr': 3e-6}})


def test_the_stateless_fp8_expert_backend_switches_at_a_resume():
    from natlang_neuralese.train.direct_sft import same_resume_identity
    saved = {'options': {'latent_lr': 1e-5}, 'inputs': {'/a': 'h'}}
    assert same_resume_identity(saved, {'options': {'latent_lr': 1e-5, 'fp8_training': 'triton-experts'},
                                        'inputs': {'/a': 'h'}})
    assert not same_resume_identity(saved, {'options': {'latent_lr': 1e-5, 'fp8_training': 'te-fp8'},
                                            'inputs': {'/a': 'h'}})


def test_fused_readout_matches_the_chunked_readout_and_the_soft_token_mixture():
    """text_warmup.fused_readout: the weighted CE, its gradient to the states, predictions and token losses equal
    chunked_readout's; the mixture equals soft_token_mixture's (same logits, computed once)."""
    import torch
    from natlang_neuralese.model.neuralese_projection import soft_token_mixture
    from natlang_neuralese.train.text_warmup import chunked_readout, fused_readout

    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(16, 40, bias=False)
            self.head.requires_grad_(False)

        def logits(self, h):
            return self.head(torch.nn.functional.rms_norm(h, (16,)))

    torch.manual_seed(0)
    backbone = Readout()
    states = torch.randn(2, 37, 16, requires_grad=True)
    targets = torch.randint(0, 40, (2, 37))
    weights = torch.rand(2, 37)
    table = torch.randn(40, 16)
    _, reference, reference_pred, _, reference_losses = chunked_readout(backbone, states, targets, 0, chunk_size=8,
                                                                        position_weights=weights)
    (reference_grad,) = torch.autograd.grad(reference * 3.0, states)
    loss, prediction, losses, mixture = fused_readout(backbone, states, targets, 0, chunk_size=8,
                                                      position_weights=weights, mixture_table=table)
    (grad,) = torch.autograd.grad(loss * 3.0, states)
    assert torch.allclose(loss, reference) and torch.allclose(grad, reference_grad, atol=1e-6)
    assert torch.equal(prediction, reference_pred) and torch.allclose(losses, reference_losses)
    assert torch.allclose(mixture, soft_token_mixture(backbone.logits, table, states.detach(), chunk=8), atol=1e-6)
    with torch.no_grad():
        assert fused_readout(backbone, states, targets, 0, position_weights=weights)[3] is None
