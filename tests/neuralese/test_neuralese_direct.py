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
        assert defaults['drift_gate_every'] >= defaults['drift_every'] and defaults['drift_warn'] < defaults['drift_gate_now'] < defaults['drift_stop']
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
