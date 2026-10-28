"""Privileged guidance: custody, the declared guidance turn, captured-block verification and whole-graph teacher scoring.

Content comes from a synthetic TRAINING-split world built here; no development or held packet row is used as
privileged content. The fixture episode has a root that imports a child's produced state and, per mode, the
in-run guidance writers (``writeGuidance`` producer turns) whose states the consumers import after their leading
system turn (``import``), only record (``write-only``, a student attempt) or replace by the text fallback (``text``).
"""
import copy
import re
import zlib
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.shared_state import SharedStateModel, concatenate_captures, native_step, splice_block
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.rl.native_replay import compile_replay
from natlang_neuralese.common.custody import CustodyError, ProtectedWorlds
from natlang_neuralese.rl.privileged_context import (REQUEST_SCHEMA, PrivilegedBlock, PrivilegedRendering,
    applicable, captured_block, case_facts, fact_request, guidance_plan, reproduces_reference, verify_captured_block)
from natlang_neuralese.rl.privileged_scoring import (_rebase_plan, action_distributions, processed_logits,
    student_form, teacher_form, teacher_role_logps, teacher_scores)
from natlang_neuralese.serve.store import native_state_id
from test_shared_state import tiny

BOS, STOP, IM_START, IM_END = 1, 2, 9, 10
TEMPERATURE = .8
CONSUMERS = {'root#turn1': ('root', 'program'), 'child#turn1': ('child', 'Count "A1"')}
WRITERS = {'root#turn1': 'guide-root#turn1', 'child#turn1': 'guide-child#turn1'}
POLICY = {'function': 'writeGuidance', 'sha256': 'a' * 64}


def encode(text):
    """Deterministic toy content tokenizer: one token per word or newline."""
    return [12 + zlib.crc32(word.encode()) % 36 for word in re.findall(r'[^\s]+|\n', text)]


def rendering(**options):
    return PrivilegedRendering('lfm2', BOS, IM_START, IM_END, 60, 61, 'f'*64, encode, **options)


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


def call(name):
    return name.rsplit('#turn', 1)[0]


def visible(name):
    """What the runtime gives the writer for one consumer: its visible facts in crisp order."""
    role, text = CONSUMERS[name]
    _, facts = case_facts(train_case(), custody=custody())
    return [fact for fact in facts if applicable(fact, {'name': call(name), 'role': role, 'text': text})]


def request(name, mode, state_id=None):
    role, _ = CONSUMERS[name]
    value = {'schema': REQUEST_SCHEMA, 'invocation': {'name': call(name), 'role': role}, 'mode': mode,
             'custody': custody().check(train_case()), 'facts': [fact_request(fact) for fact in visible(name)],
             'selection': {'implementation': 'crisp', 'sha256': 'b' * 64}}
    if mode == 'text':
        value['fallback'] = 'writer returned no native state'
    else:
        value.update(writer=POLICY, state={'id': state_id})
    return value


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
WRITER_SAMPLES = lambda roles: [(roles.produce_open, False), (3, True), (4, True), (7, True),
                                (roles.produce_close, False), (STOP, False)]
ROOT_SAMPLES = [(11, False), (13, False), (STOP, False)]


def tokens(ids, flag=False):
    return {'kind': 'tokens', 'token_ids': list(ids), 'producing': [False]*len(ids),
            **({'privileged': True} if flag else {})}


def insert(segments, at, pieces, roles, identity):
    """Actor-side insertion as the engine does it: guidance pieces at own-token index ``at``, flagged."""
    expanded = []
    for kind, value in pieces:
        if kind == 'privileged':
            expanded.append(tokens(value, True))
        else:
            expanded += [tokens([roles.consume_open], True), {'kind': 'native_state', 'id': value['id'],
                         'native_identity': identity, 'privileged': True}, tokens([roles.consume_close], True)]
    result, own, done = [], 0, False
    for segment in segments:
        if segment['kind'] != 'tokens':
            result.append(segment)
            continue
        ids = segment['token_ids']
        if not done and own <= at <= own + len(ids):
            cut = at - own
            result += [piece for piece in (tokens(ids[:cut]), *expanded, tokens(ids[cut:]))
                       if piece['kind'] != 'tokens' or piece['token_ids']]
            done = True
        else:
            result.append(segment)
        own += len(ids)
    assert done
    return result


def run_episode(model, roles, identity, mode='none', budget_tokens=512):
    """Explicit incremental cached execution of the episode (writers, child, root), with its captures."""
    device = next(model.parameters()).device
    spec = rendering(budget_tokens=budget_tokens)
    states, logps, order, captures, messages = {}, {}, [], {}, {}
    def capture(prompt):
        return {'schema': 'natlang.native-rl-sampling/1', 'policy': dict(identity), 'prompt': prompt,
                'samples': [], 'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
                'sampling_law': {'temperature': TEMPERATURE, 'seed': 3, 'guidance': False,
                                 'probabilities': 'post-processing-normalized'}}
    def feed(segments):
        out = None
        for segment in segments:
            if segment['kind'] == 'tokens':
                out = native_step(model, torch.tensor([segment['token_ids']], device=device),
                                  None if out is None else out['cache'])
            else:
                cache = splice_block(model, out['cache'], states[segment['id']],
                                     **{key: value for key, value in identity.items() if key != 'architecture'})
                out = {'cache': cache}
        return out
    def sample(name, out, record, plan, text):
        chunks, values = [], []
        for index, (token, producing) in enumerate(plan):
            logp = model.logits(out['h_final'][:, -1]).float().div(TEMPERATURE).log_softmax(-1)[0, token]
            values.append(float(logp))
            record['samples'].append({'index': index, 'token_id': token, 'logp': float(logp),
                'input_role': 'produce' if producing else 'ordinary', 'terminal': token == STOP})
            if token != STOP:
                out = native_step(model, torch.tensor([[token]], device=device), out['cache'], producing=producing)
                if producing:
                    chunks.append(out)
        logps[name] = values
        order.append(name)
        if chunks:
            block = NativeStateBlock(text=text, **identity, **concatenate_captures(chunks)).validate()
            states[native_state_id(block)] = block
            record['writes'] = [{'state_id': native_state_id(block), 'begin_sample': 0, 'end_sample': len(plan) - 1,
                                 'payload_begin': 1, 'payload_end': len(plan) - 2, 'native_identity': identity}]
            return native_state_id(block)
        return None
    def guided(name, segments):
        record = capture(segments)
        if mode == 'none':
            return segments, record
        role, text = CONSUMERS[name]
        own = [t for s in segments if s['kind'] == 'tokens' for t in s['token_ids']]
        at, pieces, receipt = guidance_plan(spec, request(name, mode, hints.get(name)), token_ids=own, text=text)
        segments = insert(segments, at, pieces, roles, identity) if pieces else segments
        record = capture(segments)
        record['privileged'] = receipt
        return segments, record
    hints = {}
    with torch.no_grad():
        if mode in ('import', 'write-only'):
            for consumer, writer in WRITERS.items():
                facts = '\n'.join(fact.text for fact in visible(consumer))
                prompt = [BOS, *turn('system', 'Write guidance.'), *turn('user', facts), IM_START,
                          *encode('assistant\n')]
                record = capture([tokens(prompt)])
                record['guidance_policy'] = dict(POLICY)
                hints[consumer] = sample(writer, feed(record['prompt']), record, WRITER_SAMPLES(roles),
                                         f'guidance for {consumer}: return await count(a1);')
                captures[writer], messages[writer] = record, [{'role': 'user', 'content': facts}]
        child_prompt, root_before, root_after = prompts(roles)
        segments, child = guided('child#turn1', [tokens(child_prompt)])
        payload = sample('child#turn1', feed(segments), child, CHILD_SAMPLES(roles), 'payload')
        segments, root = guided('root#turn1', [tokens(root_before), {'kind': 'native_state', 'id': payload,
                                'native_identity': identity}, tokens(root_after)])
        sample('root#turn1', feed(segments), root, ROOT_SAMPLES, None)
    turns = [{'graph_turn': 'root#turn1', 'sampling': root, 'read_bindings': {payload: 'write-0'},
              'messages': [{'role': 'user', 'content': 'program'}]},
             {'graph_turn': 'child#turn1', 'sampling': child, 'written_nodes': {payload: 'write-0'},
              'messages': [{'role': 'user', 'content': 'Count "A1"'}]}]
    for consumer, writer in WRITERS.items():
        if writer in captures:
            turns.append({'graph_turn': writer, 'sampling': captures[writer],
                          'written_nodes': {hints[consumer]: 'write-' + writer}, 'messages': messages[writer]})
            reader = next(item for item in turns if item['graph_turn'] == consumer)
            if mode == 'import':
                reader['read_bindings'] = {**reader.get('read_bindings', {}), hints[consumer]: 'write-' + writer}
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block=states.__getitem__)
    return SimpleNamespace(turns=turns, replay=replay, block=states[payload], states=states, logps=logps,
                           order=order, hints=hints)


def insertion_blocks(episode, mode='import', **options):
    """Guidance blocks a student episode's twin inserts: text blocks, or imports of its write-only states."""
    spec = rendering(**options)
    blocks = {}
    for name, (role, text) in CONSUMERS.items():
        node = next(node for node in episode.replay.plan.invocations if node.name == name)
        at, pieces, receipt = guidance_plan(spec, request(name, 'text' if mode == 'text' else 'write-only',
                                                          episode.hints.get(name)), token_ids=node.token_ids, text=text)
        if mode == 'text':
            blocks[name] = PrivilegedBlock(name, at, tuple(pieces[0][1]) if pieces else (), '', receipt)
        else:
            own, offset = spec.state_turn()
            blocks[name] = PrivilegedBlock(name, at, tuple(own), '', receipt, offset, episode.hints[name])
    return blocks


# ------------------------------------------------------------------ rendering, receipts and custody

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




def test_guidance_turn_is_placed_after_the_leading_system_turn_and_bound_to_its_receipt():
    spec = rendering()
    prompt = [BOS, *turn('system', 'You count.'), *turn('user', 'Count "A1"')]
    assert spec.insertion_point(prompt) == (1 + len(turn('system', 'You count.')), 'after_leading_system_turn')
    assert spec.insertion_point([BOS, *turn('user', 'x')]) == (1, 'after_bos')
    assert spec.insertion_point(turn('user', 'x')) == (0, 'start')
    at, pieces, receipt = guidance_plan(spec, request('child#turn1', 'import', 'nzs1_state'), token_ids=prompt,
                                        text='Count "A1"')
    head, tail = spec.state_parts()
    assert at == 1 + len(turn('system', 'You count.'))
    assert pieces == [('privileged', head), ('privileged_state', {'id': 'nzs1_state'}), ('privileged', tail)]
    assert [fact['id'] for fact in receipt['facts']] == ['child_value:0']  # A1's own value, not B7's
    own, offset = spec.state_turn()
    assert own[offset - 1] == spec.consume_open and own[offset] == spec.consume_close
    assert receipt['mode'] == 'import' and receipt['state'] == {'id': 'nzs1_state'} and receipt['token_count'] == len(own)
    at, pieces, receipt = guidance_plan(spec, request('child#turn1', 'write-only', 'nzs1_state'), token_ids=prompt,
                                        text='Count "A1"')
    assert pieces == [] and receipt['mode'] == 'write-only'
    at, pieces, receipt = guidance_plan(spec, request('root#turn1', 'text'), token_ids=prompt, text='program')
    assert [fact['kind'] for fact in receipt['facts']] == ['expected_result', 'reference_solution', 'minimum_sequence']
    assert pieces[0][1] == spec.turn([fact.text for fact in visible('root#turn1')])
    assert receipt['fallback'] == 'writer returned no native state'
    # The engine refuses a fact the consuming invocation may not see (visibility is an exact data rule).
    leaky = request('root#turn1', 'import', 'nzs1_state')
    leaky['facts'].append(fact_request(next(f for f in case_facts(train_case(), custody=custody())[1]
                                            if f.kind == 'child_value')))
    with pytest.raises(CustodyError, match='not visible'):
        guidance_plan(spec, leaky, token_ids=prompt, text='program')
    for broken in ({**request('root#turn1', 'import', 'x'), 'state': None},
                   {**request('root#turn1', 'text'), 'fallback': ''},
                   {**request('root#turn1', 'import', 'x'), 'custody': {'split': 'test'}},
                   {**request('root#turn1', 'import', 'x'), 'invocation': {'name': 'r'}}):
        with pytest.raises(ValueError):
            guidance_plan(spec, broken, token_ids=prompt, text='program')


def test_budget_packs_whole_facts_of_the_text_fallback_in_policy_order():
    facts = visible('root#turn1')
    one = len(rendering().turn([facts[0].text]))
    prompt = [BOS]
    _, pieces, receipt = guidance_plan(rendering(budget_tokens=one), request('root#turn1', 'text'),
                                       token_ids=prompt, text='program')
    assert [fact['id'] for fact in receipt['facts']] == [facts[0].id] and len(pieces[0][1]) <= one
    assert [item['reason'] for item in receipt['omitted']] == ['token_budget'] * 2
    _, pieces, receipt = guidance_plan(rendering(budget_tokens=3), request('root#turn1', 'text'),
                                       token_ids=prompt, text='program')
    assert pieces == [] and receipt['token_count'] == 0 and len(receipt['omitted']) == 3


@pytest.mark.parametrize('mode', ['import', 'text', 'write-only'])
def test_captured_guidance_is_verified_against_custody_facts_visibility_and_rendering(mode):
    model, roles, identity = load('lfm')
    episode = run_episode(model, roles, identity, mode)
    receipt, facts = case_facts(train_case(), custody=custody())
    for name, (role, text) in CONSUMERS.items():
        item = next(turn for turn in episode.turns if turn['graph_turn'] == name)
        block = captured_block(item)
        assert block.receipt['mode'] == mode and (block.state_id is not None) == (mode != 'text')
        invocation = {'name': call(name), 'role': role, 'text': text}
        used = verify_captured_block(block, rendering(), custody_receipt=receipt, facts=facts, invocation=invocation)
        assert [fact.id for fact in used] == [fact.id for fact in visible(name)]
        if mode == 'write-only':
            continue
        tampered = PrivilegedBlock(name, block.insert_at, block.token_ids[:-1], '', block.receipt,
                                   block.import_offset, block.state_id)
        with pytest.raises(ValueError):
            verify_captured_block(tampered, rendering(), custody_receipt=receipt, facts=facts, invocation=invocation)
        with pytest.raises(ValueError):  # another rendering declaration is another hash
            verify_captured_block(block, rendering(header='Other header:'), custody_receipt=receipt, facts=facts,
                                  invocation=invocation)
        with pytest.raises(CustodyError):  # the world's custody changed since the attempt
            verify_captured_block(block, rendering(), custody_receipt={**receipt, 'protected_sha256': '9'*64},
                                  facts=facts, invocation=invocation)
        with pytest.raises(CustodyError):  # visibility is re-verified against the graph's role
            verify_captured_block(block, rendering(), custody_receipt=receipt, facts=facts,
                                  invocation={**invocation, 'role': 'child' if role == 'root' else 'root'})
    unflagged = copy.deepcopy(episode.turns[0])
    unflagged['sampling'].pop('privileged')
    if mode != 'write-only':
        with pytest.raises(ValueError, match='receipt'):
            captured_block(unflagged)


def test_verbatim_reference_telemetry():
    facts = visible('root#turn1')
    assert reproduces_reference('Use return await count(a1); and finish.', facts) is True  # a reference line
    assert reproduces_reference('The result is 2', facts) is False  # one-character values are not counted
    assert reproduces_reference('count the apples carefully', visible('child#turn1')) is False


def test_graph_edit_refuses_ambiguous_boundaries_and_moves_guidance_imports():
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
    # An inserted guidance turn carries its own import of a writer's span; removing it drops exactly that import.
    guided, _ = _rebase_plan(plan, {'b': _Edit(0, 0, (60, 61), (ImportSpan(1, 'a', 2, 4),))})
    assert guided.invocations[1].imports == (ImportSpan(1, 'a', 2, 4), ImportSpan(3, 'a', 2, 4))
    restored, _ = _rebase_plan(guided, {'b': _Edit(0, 2, (), (), (1,))})
    assert restored == plan
    with pytest.raises(ValueError, match='imports differ'):
        _rebase_plan(guided, {'b': _Edit(0, 2, ())})
    with pytest.raises(ValueError, match='another invocation'):
        _rebase_plan(plan, {'b': _Edit(0, 0, (60, 61), (ImportSpan(1, 'b', 0, 1),))})


# ------------------------------------------------------------------ teacher scoring on the whole graph

@pytest.mark.parametrize('family,mode', [('lfm', 'import'), ('mellum', 'import'), ('lfm3', 'import'), ('lfm', 'text')])
def test_teacher_scores_equal_explicit_cached_execution_with_imports(family, mode):
    model, roles, identity = load(family)
    student = run_episode(model, roles, identity, 'write-only' if mode == 'import' else 'none')
    blocks = insertion_blocks(student, mode)
    explicit = run_episode(model, roles, identity, mode)  # teacher-role cached execution
    scores = teacher_scores(model, student.replay, blocks, verify_reference=True)
    # Alignment: one teacher score per student-sampled token, same order and tokens.
    assert scores.actions == student.replay.actions
    expected = torch.tensor([explicit.logps[action['turn']][action['sample']] for action in student.replay.actions])
    torch.testing.assert_close(scores.logps.cpu(), expected, rtol=2e-5, atol=2e-5)
    assert not scores.logps.requires_grad and scores.logits.dtype == torch.float32
    consumers = [a for a in student.replay.actions if a['turn'] in CONSUMERS]
    student_logps = torch.tensor([student.logps[a['turn']][a['sample']] for a in consumers])
    teacher_logps = torch.tensor([explicit.logps[a['turn']][a['sample']] for a in consumers])
    assert (teacher_logps - student_logps).abs().max() > 1e-4  # guidance changes the teacher
    teacher = teacher_form(student.replay, blocks)
    old = {node.name: node for node in student.replay.plan.invocations}
    new = {node.name: node for node in teacher.plan.invocations}
    shift = {name: len(block.token_ids) for name, block in blocks.items()}
    assert new['root#turn1'].imports[-1].at == old['root#turn1'].imports[-1].at + shift['root#turn1']
    assert new['child#turn1'].producing == tuple(tuple(v + shift['child#turn1'] for v in span)
                                                 for span in old['child#turn1'].producing)
    if mode == 'import':
        for name in CONSUMERS:
            imported = [edge for edge in new[name].imports if edge.source == WRITERS[name]]
            assert [edge.at for edge in imported] == [blocks[name].insert_at + blocks[name].import_offset]
    # Round trip: removing the verified blocks restores the student graph exactly.
    # (An inserted import may move its writer earlier in the plan; nodes, imports and local queries are exact.)
    restored = student_form(teacher, blocks)
    shape = lambda plan: {n.name: (n.token_ids, n.imports, n.producing) for n in plan.invocations}
    assert shape(restored.plan) == shape(student.replay.plan)
    local = lambda replay: [(a['turn'], a['query'] - dict(zip((n.name for n in replay.plan.invocations),
                                                              replay.plan.offsets))[a['turn']]) for a in replay.actions]
    assert local(restored) == local(student.replay)
    # The explicit teacher-role capture compiles to the same teacher graph.
    assert ({node.name: (node.token_ids, node.imports) for node in explicit.replay.plan.invocations}
            == {node.name: (node.token_ids, node.imports) for node in teacher.plan.invocations})


def test_empty_blocks_score_the_student_policy_itself():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity)
    blocks = insertion_blocks(student, 'text', budget_tokens=3)
    assert not any(block.token_ids for block in blocks.values())
    scores = teacher_scores(model, student.replay, blocks)
    behavior = torch.tensor([action['behavior_logp'] for action in student.replay.actions])
    torch.testing.assert_close(scores.logps.cpu(), behavior, rtol=2e-5, atol=2e-5)


def test_topk_returns_processed_logits_and_full_normalizer():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity, 'write-only')
    blocks = insertion_blocks(student)
    full = teacher_scores(model, student.replay, blocks)
    top = teacher_scores(model, student.replay, blocks, topk=5)
    assert top.logits is None and top.topk_ids.shape == (len(student.replay.actions), 5)
    for row, action in enumerate(student.replay.actions):
        processed = processed_logits(full.logits[row], action)
        logp = processed - torch.logsumexp(processed, -1)
        torch.testing.assert_close(top.topk_logits[row] - top.log_normalizer[row], logp[top.topk_ids[row]])
    torch.testing.assert_close(top.logps, full.logps)


def test_teacher_role_attempts_train_the_writer_through_the_import_and_have_an_exact_student_form():
    model, roles, identity = load('lfm')
    attempt = run_episode(model, roles, identity, 'import')  # sampled WITH the imported guidance
    blocks = {name: captured_block(item) for name, item in
              ((item['graph_turn'], item) for item in attempt.turns) if name in CONSUMERS}
    student = run_episode(model, roles, identity, 'write-only')
    logps, behavior, student_view = teacher_role_logps(model, attempt.replay, blocks)
    torch.testing.assert_close(logps.detach(), behavior, rtol=2e-5, atol=2e-5)
    view = {node.name: (node.token_ids, node.imports) for node in student_view.plan.invocations}
    assert view == {node.name: (node.token_ids, node.imports) for node in student.replay.plan.invocations}
    # The consumers' outcome reaches the writer's sampled payload through the producer/consumer import.
    consumer = torch.tensor([a['turn'] in CONSUMERS for a in attempt.replay.actions])
    embedding = next(p for name, p in model.named_parameters() if 'embed' in name and p.shape[0] == 64)
    grad = torch.autograd.grad(-logps[consumer].sum(), [model.adapter.up.weight, embedding])
    assert grad[0].abs().sum() > 0 and grad[1][[3, 4, 7]].abs().sum() > 0  # rows only the writer's payload uses
    wrong = {**blocks, 'root#turn1': PrivilegedBlock('root#turn1', blocks['root#turn1'].insert_at,
                                                     (IM_START,) * len(blocks['root#turn1'].token_ids), '', {},
                                                     blocks['root#turn1'].import_offset, blocks['root#turn1'].state_id)}
    with pytest.raises(ValueError, match='declared privileged block'):
        student_form(attempt.replay, wrong)


def test_reverse_kl_opsd_term_from_teacher_logits_has_finite_gradients():
    model, roles, identity = load('lfm')
    student = run_episode(model, roles, identity, 'write-only')
    blocks = insertion_blocks(student)
    teacher = teacher_scores(model, student.replay, blocks)
    current = action_distributions(model, student.replay)  # student role, with gradients
    assert current.logits.requires_grad
    terms = []
    for row, action in enumerate(student.replay.actions):
        if action['turn'] not in CONSUMERS:
            continue
        log_s = torch.log_softmax(processed_logits(current.logits[row], action), -1)
        log_t = torch.log_softmax(processed_logits(teacher.logits[row], action), -1)
        terms.append((log_s.exp() * (log_s - log_t)).sum())
    kl = torch.stack(terms).mean()
    assert torch.isfinite(kl) and kl > 0
    kl.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0  # producer receives the consumer's OPSD signal
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
