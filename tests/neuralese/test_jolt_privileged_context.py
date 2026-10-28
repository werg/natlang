"""JOLT+ privileged context: custody, rendering, and whole-graph teacher scoring.

Content comes from a synthetic TRAINING-split world built here; no development
or held packet row is used as privileged content.
"""
import re
import sys
import zlib
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.shared_state import SharedStateModel, concatenate_captures, native_step, splice_block
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.rl.native_replay import compile_replay
from natlang_neuralese.common.custody import CustodyError, ProtectedWorlds
from natlang_neuralese.rl.privileged_context import (PrivilegedBlock, PrivilegedRendering, case_facts,
    insert_into_prompt_segments, privileged_blocks, render_block, verify_block)
from natlang_neuralese.rl.privileged_scoring import (_rebase_plan, action_distributions, graph_invocations,
    processed_logits, student_form, teacher_form, teacher_role_logps, teacher_scores)
from natlang_neuralese.serve.store import native_state_id
from test_shared_state import tiny

BOS, STOP, IM_START, IM_END = 1, 2, 9, 10
TEMPERATURE = .8


def encode(text):
    """Deterministic toy content tokenizer: one token per word or newline."""
    return [12 + zlib.crc32(word.encode()) % 36 for word in re.findall(r'[^\s]+|\n', text)]


def rendering(**options):
    return PrivilegedRendering('lfm2', BOS, IM_START, IM_END, 'f'*64, encode, **options)


def turn(role, content):
    return [IM_START, *encode(f'{role}\n'), *encode(content), IM_END, *encode('\n')]


def train_case(**overrides):
    case = {'id': 'jolt-fixture:train:0', 'split': 'train', 'source_ids': ['jolt-fixture:train:0'],
            'source_groups': ['jolt-fixture:g0'], 'family': 'fixture_count',
            'curriculum': {'split_group': 'jolt-fixture:g0', 'pair_group': 'jolt-fixture:p0',
                'evidence': {'world': ['A1: two apples']},
                'minimum_sequence': ['ask the child', 'return its count'],
                'reference': {'root': [['eval', {'code': 'return await count(a1);'}],
                                       ['return_result', {'status': 'success', 'value': 2}]],
                              'children': [{'match': ['Count', '"A1"'], 'value': 2},
                                           {'match': ['Count', '"B7"'], 'value': 5}]}},
            'semantics': {'root': 'count.nl', 'files': {'count.nl': 'Count the apples.'}, 'expected': 2}}
    case.update(overrides)
    return case


def protected_case():
    return {'id': 'jolt-fixture:dev:0', 'split': 'test', 'source_ids': ['jolt-fixture:dev:0'],
            'source_groups': ['jolt-fixture:dev-g0'],
            'curriculum': {'split_group': 'jolt-fixture:dev-g0', 'pair_group': 'jolt-fixture:dev-p0',
                           'evidence': {'world': ['Z9: nine pears']}},
            'semantics': {'root': 'pears.nl', 'files': {'pears.nl': 'Count the pears.'}, 'expected': 9}}


def custody():
    return ProtectedWorlds.from_cases({'jolt-dev-fixture': ('0'*64, [protected_case()])})


# ------------------------------------------------------------------ episodes

def load(family):
    torch.manual_seed(901)
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    with torch.no_grad():
        model.adapter.up.weight.normal_(0, .05)  # nonzero producer transform
    roles = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
    identity = {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
                'embedding_weight_sha256': '3'*64, 'architecture': model.port.config.model_type}
    return model, roles, identity


def prompts(roles):
    child = [BOS, *turn('system', 'You count.'), *turn('user', 'Count "A1"'), IM_START, *encode('assistant\n')]
    root_before = [BOS, *turn('system', 'You run programs.'), IM_START, *encode('user\nprogram'), roles.consume_open]
    root_after = [roles.consume_close, IM_END, *encode('\n'), IM_START, *encode('assistant\n')]
    return child, root_before, root_after


CHILD_SAMPLES = lambda roles: [(roles.produce_open, False), (5, True), (6, True), (roles.produce_close, False), (STOP, False)]
ROOT_SAMPLES = [(11, False), (13, False), (STOP, False)]


def run_episode(model, roles, identity, blocks=None):
    """Explicit incremental cached execution of the two-invocation graph.

    With ``blocks`` the prompts carry the privileged turns (teacher role),
    inserted by the same actor-side helper the sampler uses.
    """
    blocks = blocks or {}
    device = next(model.parameters()).device
    child_prompt, root_before, root_after = prompts(roles)
    def capture(prompt):
        return {'schema': 'natlang.native-rl-sampling/1', 'policy': dict(identity), 'prompt': prompt,
                'samples': [], 'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
                'sampling_law': {'temperature': TEMPERATURE, 'seed': 3, 'guidance': False,
                                 'probabilities': 'post-processing-normalized'}}
    def feed(segments, splice=None):
        out = None
        for segment in segments:
            if segment['kind'] == 'tokens':
                out = native_step(model, torch.tensor([segment['token_ids']], device=device),
                                  None if out is None else out['cache'])
            else:
                cache = splice_block(model, out['cache'], splice,
                                     **{key: value for key, value in identity.items() if key != 'architecture'})
                out = {'cache': cache}
        return out
    def sample(out, record, tokens):
        chunks, logps = [], []
        for index, (token, producing) in enumerate(tokens):
            logp = model.logits(out['h_final'][:, -1]).float().div(TEMPERATURE).log_softmax(-1)[0, token]
            logps.append(float(logp))
            record['samples'].append({'index': index, 'token_id': token, 'logp': float(logp),
                'input_role': 'produce' if producing else 'ordinary', 'terminal': token == STOP})
            if token != STOP:
                out = native_step(model, torch.tensor([[token]], device=device), out['cache'], producing=producing)
                if producing:
                    chunks.append(out)
        return chunks, logps
    with torch.no_grad():
        segments = [{'kind': 'tokens', 'token_ids': child_prompt, 'producing': [False]*len(child_prompt)}]
        if 'child#turn1' in blocks:
            segments = insert_into_prompt_segments(segments, blocks['child#turn1'])
        child = capture(segments)
        chunks, child_logps = sample(feed(segments), child, CHILD_SAMPLES(roles))
        block = NativeStateBlock(text='payload', **identity, **concatenate_captures(chunks)).validate()
        block_id = native_state_id(block)
        child['writes'] = [{'state_id': block_id, 'begin_sample': 0, 'end_sample': 4,
                            'payload_begin': 1, 'payload_end': 3, 'native_identity': identity}]
        segments = [{'kind': 'tokens', 'token_ids': root_before, 'producing': [False]*len(root_before)},
                    {'kind': 'native_state', 'id': block_id, 'native_identity': identity},
                    {'kind': 'tokens', 'token_ids': root_after, 'producing': [False]*len(root_after)}]
        if 'root#turn1' in blocks:
            segments = insert_into_prompt_segments(segments, blocks['root#turn1'])
        root = capture(segments)
        out = feed(segments, splice=block)
        # A splice returns only the cache; the next token segment supplies h_final.
        _, root_logps = sample(out, root, ROOT_SAMPLES)
    turns = [{'graph_turn': 'root#turn1', 'sampling': root, 'read_bindings': {block_id: 'write-0'},
              'messages': [{'role': 'user', 'content': 'program'}]},
             {'graph_turn': 'child#turn1', 'sampling': child, 'written_nodes': {block_id: 'write-0'},
              'messages': [{'role': 'user', 'content': 'Count "A1"'}]}]
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block={block_id: block}.__getitem__)
    return SimpleNamespace(turns=turns, replay=replay, block=block,
                           logps={'child#turn1': child_logps, 'root#turn1': root_logps})


def episode_blocks(replay, turns, **options):
    return privileged_blocks(rendering(**options), train_case(), graph_invocations(replay, turns, root_calls={'root'}),
                             custody=custody())


# ------------------------------------------------------------------ rendering and custody

def test_rendering_places_block_after_leading_system_turn_and_binds_receipt():
    spec = rendering()
    prompt = [BOS, *turn('system', 'You count.'), *turn('user', 'Count "A1"')]
    assert spec.insertion_point(prompt) == (1 + len(turn('system', 'You count.')), 'after_leading_system_turn')
    assert spec.insertion_point([BOS, *turn('user', 'x')]) == (1, 'after_bos')
    assert spec.insertion_point(turn('user', 'x')) == (0, 'start')
    receipt, facts = case_facts(train_case(), custody=custody())
    block = render_block(spec, invocation={'name': 'c', 'role': 'child', 'text': 'Count "A1" now'},
                         token_ids=prompt, facts=facts, custody_receipt=receipt)
    assert [fact['kind'] for fact in block.receipt['facts']] == ['child_value']
    assert block.token_ids[0] == IM_START and block.token_ids[-len(encode('\n'))-1] == IM_END
    assert 'answer of this call is 2' in block.text  # the matching child, not B7's 5
    verify_block(block, spec, custody=custody())
    tampered = PrivilegedBlock(block.invocation, block.insert_at, block.token_ids[:-1], block.text, block.receipt)
    with pytest.raises(CustodyError):
        verify_block(tampered, spec, custody=custody())
    with pytest.raises(CustodyError):  # another rendering declaration is another hash
        verify_block(block, rendering(header='Other header:'), custody=custody())
    root = render_block(spec, invocation={'name': 'r', 'role': 'root', 'text': 'program'},
                        token_ids=prompt, facts=facts, custody_receipt=receipt)
    assert [fact['kind'] for fact in root.receipt['facts']] == ['expected_result', 'reference_solution',
                                                                'minimum_sequence']
    assert root.receipt['policy']['implementation'] == 'crisp'


def test_custody_refuses_development_confirmation_and_related_worlds(tmp_path):
    protected = custody()
    assert protected.check(train_case())['split'] == 'train'
    for split in ('test', 'heldout', 'development', 'confirmation', None):
        with pytest.raises(CustodyError, match='split'):
            protected.check(train_case(split=split))
    related = train_case()
    related['curriculum'] = {**related['curriculum'], 'pair_group': 'jolt-fixture:dev-p0'}
    with pytest.raises(CustodyError, match='protected world'):
        protected.check(related)
    with pytest.raises(CustodyError, match='protected world'):
        protected.check(train_case(source_groups=['jolt-fixture:dev-g0']))
    with pytest.raises(CustodyError, match='protected world'):
        protected.check(train_case(id='jolt-fixture:dev:0'))
    relabelled = protected_case()  # same world content under fresh train labels
    relabelled.update(id='fresh', split='train', source_ids=['fresh'], source_groups=['fresh-g'])
    relabelled['curriculum'] = {**relabelled['curriculum'], 'split_group': 'fresh-g', 'pair_group': 'fresh-p'}
    with pytest.raises(CustodyError, match='world'):
        protected.check(relabelled)
    with pytest.raises(CustodyError):  # facts are never derived from a refused world
        case_facts(train_case(split='test'), custody=protected)
    with pytest.raises(CustodyError, match='trajectories'):
        case_facts(train_case(), custody=protected, teacher_trajectories=[
            {'case_id': 'jolt-fixture:train:0', 'split': 'train', 'verified': False, 'actor': 'qwen', 'actions': []}])
    with pytest.raises(CustodyError, match='earlier attempts'):
        case_facts(train_case(), custody=protected, attempt_feedback=[
            {'case_id': 'jolt-fixture:dev:0', 'attempt': 1, 'feedback': 'x'}])
    facts = case_facts(train_case(), custody=protected, teacher_trajectories=[
        {'case_id': 'jolt-fixture:train:0', 'split': 'train', 'verified': True, 'actor': 'qwen3.6',
         'actions': [['eval', {'code': 'return 2;'}]]}],
        attempt_feedback=[{'case_id': 'jolt-fixture:train:0', 'attempt': 1, 'feedback': 'TypeError: x'}])[1]
    assert {'teacher_trajectory', 'attempt_feedback'} <= {fact.kind for fact in facts}


def test_budget_packs_whole_facts_in_policy_order():
    receipt, facts = case_facts(train_case(), custody=custody())
    invocation = {'name': 'r', 'role': 'root', 'text': 'program'}
    full = render_block(rendering(budget_tokens=4096), invocation=invocation, token_ids=[BOS], facts=facts,
                        custody_receipt=receipt)
    one = len(rendering().turn([facts[0].text]))
    small = render_block(rendering(budget_tokens=one), invocation=invocation, token_ids=[BOS], facts=facts,
                         custody_receipt=receipt)
    assert [fact['id'] for fact in small.receipt['facts']] == [full.receipt['facts'][0]['id']]
    assert [item['reason'] for item in small.receipt['omitted']] == ['token_budget'] * 2
    assert len(small.token_ids) <= one
    empty = render_block(rendering(budget_tokens=3), invocation=invocation, token_ids=[BOS], facts=facts,
                         custody_receipt=receipt)
    assert empty.token_ids == () and empty.text == '' and len(empty.receipt['omitted']) == 3


def test_nl_policy_command_selects_and_is_validated():
    receipt, facts = case_facts(train_case(), custody=custody())
    invocation = {'name': 'r', 'role': 'root', 'text': 'program'}
    reverse = [sys.executable, '-c', 'import json,sys; r=json.load(sys.stdin); '
               'print(json.dumps([f["id"] for f in r["facts"]][::-1]))']
    block = render_block(rendering(), invocation=invocation, token_ids=[BOS], facts=facts, custody_receipt=receipt,
                         command=reverse)
    assert [fact['kind'] for fact in block.receipt['facts']] == ['minimum_sequence', 'reference_solution',
                                                                'expected_result']
    assert block.receipt['policy']['implementation'] == 'nl' and len(block.receipt['policy']['sha256']) == 64
    leaky = [sys.executable, '-c', 'print(\'["child_value:0"]\')']  # not applicable to the root call
    with pytest.raises(ValueError, match='applicable'):
        render_block(rendering(), invocation=invocation, token_ids=[BOS], facts=facts, custody_receipt=receipt,
                     command=leaky)


def test_graph_edit_refuses_ambiguous_boundaries():
    plan = compile_graph([Invocation('a', (1, 2, 3, 4, 5), producing=((2, 4),)),
                          Invocation('b', (6, 7, 8), imports=(ImportSpan(1, 'a', 2, 4),))])
    from natlang_neuralese.rl.privileged_scoring import _Edit
    with pytest.raises(ValueError, match='crosses'):
        _rebase_plan(plan, {'a': _Edit(3, 0, (9, 9))})
    with pytest.raises(ValueError, match='import boundary'):
        _rebase_plan(plan, {'b': _Edit(1, 0, (9,))})
    rebased, _ = _rebase_plan(plan, {'a': _Edit(1, 0, (9, 9)), 'b': _Edit(0, 0, (9,))})
    assert rebased.invocations[0].producing == ((4, 6),)
    assert rebased.invocations[1].imports[0] == ImportSpan(2, 'a', 4, 6)


# ------------------------------------------------------------------ teacher scoring on the whole graph

@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_teacher_scores_equal_explicit_cached_execution_with_imports(family):
    model, roles, identity = load(family)
    student = run_episode(model, roles, identity)
    blocks = episode_blocks(student.replay, student.turns)
    assert all(block.token_ids for block in blocks.values())
    explicit = run_episode(model, roles, identity, blocks=blocks)  # teacher-role cached execution
    scores = teacher_scores(model, student.replay, blocks, verify_reference=True)
    # Alignment: one teacher score per student-sampled token, same order and tokens.
    assert scores.actions == student.replay.actions
    assert scores.token_ids.tolist() == [action['token_id'] for action in student.replay.actions]
    expected = torch.tensor([explicit.logps[action['turn']][action['sample']] for action in student.replay.actions])
    torch.testing.assert_close(scores.logps.cpu(), expected, rtol=2e-5, atol=2e-5)
    assert not scores.logps.requires_grad and scores.logits.dtype == torch.float32
    student_logps = torch.tensor([student.logps[action['turn']][action['sample']]
                                  for action in student.replay.actions])
    assert (scores.logps.cpu() - student_logps).abs().max() > 1e-4  # privileged context changes the teacher
    # Geometry: the consumer import moved by the consumer's block; the imported
    # span by the producer's block; producer spans likewise.
    teacher = teacher_form(student.replay, blocks)
    shift = {name: len(block.token_ids) for name, block in blocks.items()}
    old = {node.name: node for node in student.replay.plan.invocations}
    new = {node.name: node for node in teacher.plan.invocations}
    assert new['root#turn1'].imports[0].at == old['root#turn1'].imports[0].at + shift['root#turn1']
    assert (new['root#turn1'].imports[0].start, new['root#turn1'].imports[0].stop) == tuple(
        value + shift['child#turn1'] for value in (old['root#turn1'].imports[0].start, old['root#turn1'].imports[0].stop))
    assert new['child#turn1'].producing == tuple(tuple(v + shift['child#turn1'] for v in span)
                                                 for span in old['child#turn1'].producing)
    # Round trip: removing the verified blocks restores the student graph exactly.
    restored = student_form(teacher, blocks)
    assert restored.plan == student.replay.plan
    assert [a['query'] for a in restored.actions] == [a['query'] for a in student.replay.actions]
    # The explicit teacher-role capture compiles to the same teacher graph.
    assert [node.token_ids for node in explicit.replay.plan.invocations] == [
        node.token_ids for node in teacher.plan.invocations]


def test_empty_blocks_score_the_student_policy_itself():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity)
    blocks = episode_blocks(student.replay, student.turns, budget_tokens=3)
    assert not any(block.token_ids for block in blocks.values())
    scores = teacher_scores(model, student.replay, blocks)
    behavior = torch.tensor([action['behavior_logp'] for action in student.replay.actions])
    torch.testing.assert_close(scores.logps.cpu(), behavior, rtol=2e-5, atol=2e-5)


def test_topk_returns_processed_logits_and_full_normalizer():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity)
    blocks = episode_blocks(student.replay, student.turns)
    full = teacher_scores(model, student.replay, blocks)
    top = teacher_scores(model, student.replay, blocks, topk=5)
    assert top.logits is None and top.topk_ids.shape == (len(student.replay.actions), 5)
    for row, action in enumerate(student.replay.actions):
        processed = processed_logits(full.logits[row], action)
        logp = processed - torch.logsumexp(processed, -1)
        torch.testing.assert_close(top.topk_logits[row] - top.log_normalizer[row], logp[top.topk_ids[row]])
    torch.testing.assert_close(top.logps, full.logps)


def test_teacher_role_attempts_have_gradients_and_exact_student_form():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity)
    blocks = episode_blocks(student.replay, student.turns)
    attempt = run_episode(model, roles, identity, blocks=blocks)  # sampled WITH the privileged blocks
    logps, behavior, student_view = teacher_role_logps(model, attempt.replay, blocks)
    torch.testing.assert_close(logps.detach(), behavior, rtol=2e-5, atol=2e-5)
    assert [node.token_ids for node in student_view.plan.invocations] == [
        node.token_ids for node in student.replay.plan.invocations]
    assert [node.imports for node in student_view.plan.invocations] == [
        node.imports for node in student.replay.plan.invocations]
    (-logps.mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert model.adapter.up.weight.grad.abs().sum() > 0  # through the teacher-role producer
    assert all(torch.isfinite(grad).all() for grad in grads)
    wrong = {**blocks, 'root#turn1': PrivilegedBlock('root#turn1', blocks['root#turn1'].insert_at,
                                                     (IM_START,) * len(blocks['root#turn1'].token_ids), '', {})}
    with pytest.raises(ValueError, match='declared privileged block'):
        student_form(attempt.replay, wrong)


def test_reverse_kl_opsd_term_from_teacher_logits_has_finite_gradients():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity)
    blocks = episode_blocks(student.replay, student.turns)
    teacher = teacher_scores(model, student.replay, blocks)
    current = action_distributions(model, student.replay)  # student role, with gradients
    assert current.logits.requires_grad
    terms = []
    for row, action in enumerate(student.replay.actions):
        log_s = torch.log_softmax(processed_logits(current.logits[row], action), -1)
        log_t = torch.log_softmax(processed_logits(teacher.logits[row], action), -1)
        terms.append((log_s.exp() * (log_s - log_t)).sum())
    kl = torch.stack(terms).mean()
    assert torch.isfinite(kl) and kl > 0
    kl.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0  # producer receives the consumer's OPSD signal
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
               if p.grad is not None and not name.startswith('adapter.')) > 0
