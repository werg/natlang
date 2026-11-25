"""Engine A call-level extraction and teaching (scripts/neuralese_data/call_level_engine.py) on real-shaped rollouts."""
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'scripts/neuralese_data', ROOT / 'training/neuralese', ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import call_level_engine as cle  # noqa: E402
import sft_sources as S  # noqa: E402

SYSTEM_TOOLS = 'Answer with the return_result tool: call it with status "success" and a value of the declared type.'
SYSTEM_READOUT = 'You are running one call of a natural-language function. Your reply is the function\'s return value.'
TOOLS = [{'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {'type': 'object'}}}
         for n in ('eval', 'read_code', 'return_result')]


def scope(args_text):
    """The host's argument scope: a host-generated eval call and its tool output (as aei1 rollouts record it)."""
    return [{'role': 'assistant', 'content': '', 'natlang_host_generated': True,
             'reasoning_content': "I'll start by reading this call's arguments into the eval scope.",
             'tool_calls': [{'id': 'scope_0', 'type': 'function',
                             'function': {'name': 'eval', 'arguments': '{"code":"const inputs = read_inputs();"}'}}]},
            {'role': 'tool', 'tool_call_id': 'scope_0', 'content': args_text}]


def root_turn(call_id, action):
    opening = ('You are inside this call: retime_board(): RetimeBoardResult\n'
               'type RetimeBoardResult = { notices: number; cancelled: number; live: number };\n\n'
               'Instructions:\nApply the change notices.')
    return {'messages': [{'role': 'system', 'content': SYSTEM_TOOLS}, {'role': 'user', 'content': opening},
                         *scope('ok')],
            'tools': TOOLS, 'target': {'role': 'assistant', 'content': '', 'reasoning_content': 'read first',
                                       'tool_calls': [{'id': 'c0', 'type': 'function', 'function': action}]},
            'invocation_id': call_id, 'function': 'unknown', 'definition_id': None, 'parent_invocation': None,
            'invocation_role': 'root',
            'sampling': {'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3]}],
                         'samples': [{'token_id': 7}, {'token_id': 8}]}}


def step_turn(call_id):
    turn = root_turn(call_id, {'name': 'return_result',
                               'arguments': '{"status":"success","value":{"notices":2,"cancelled":1,"live":3}}'})
    turn['messages'] = turn['messages'] + [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c0', 'type': 'function',
                                                             'function': {'name': 'eval', 'arguments': '{"code":"1"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c0', 'content': '1'}]
    return turn


def decision_turn(call_id, parent, evidence, answer):
    opening = ('You are inside this call: verify(evidence: string): "pass" | "fail" | "needs-review"\n\n'
               'Instructions:\nPass only when the name is preserved.')
    return {'messages': [{'role': 'system', 'content': SYSTEM_READOUT}, {'role': 'user', 'content': opening},
                         *scope(f'evidence: string = "{evidence}"'),
                         {'role': 'user', 'content': 'Answer this call now, without tools. Reply with exactly one of '
                                                     'these JSON values and nothing else: "pass", "fail", "needs-review".'}],
            'tools': None, 'target': {'role': 'assistant', 'content': json.dumps(answer), 'reasoning_content': 'hmm'},
            'invocation_id': call_id, 'function': 'unknown', 'definition_id': None, 'parent_invocation': parent,
            'invocation_role': 'child',
            'decision_reasoning': {'index': 0, 'samples': 4, 'reasoning_sample_ids': [1, 2, 3]}}


def invocation(call_id, definition, signature):
    return {'version': 'reduction-trace/1', 'kind': 'invocation', 'phase': 'start', 'call_id': call_id,
            'definition': {'id': definition, 'revision': ''},
            'native_occurrence': {'definition_id': definition, 'signature': signature,
                                  'argument_binding_sha256': 'ab' * 32}}


def rollout(task_id, sample, outcome, decision_answer, evidence='Mara'):
    root, child = f'ep{sample}/root', f'ep{sample}/child'
    return {'schema': 'natlang.rollout/1', 'task_id': task_id, 'family': 'auth_x', 'sample': sample, 'split': 'train',
            'source_groups': ['authored:x'], 'outcome': outcome,
            'reward': {'quality': 1.0 if outcome == 'task_success' else 0.0},
            'policy': {'checkpoint_sha256': 'c' * 64},
            'turns': [root_turn(root, {'name': 'eval', 'arguments': '{"code":"read()"}'}),
                      decision_turn(child, root, evidence, decision_answer), step_turn(root)],
            'execution_graph': [invocation(root, 'retime_board', '() => RetimeBoardResult'),
                                invocation(child, 'verify', '(evidence: string) => "pass" | "fail" | "needs-review"')]}


def write_group(coll, name, task_id, rollouts, split='train'):
    g = coll / 'groups' / name
    g.mkdir(parents=True)
    (g / 'cases.jsonl').write_text(json.dumps({'id': task_id, 'split': split, 'source_groups': ['authored:x']}) + '\n')
    (g / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rollouts))
    (g / 'score.json').write_text('{}')


@pytest.fixture
def collection(tmp_path, monkeypatch):
    coll = tmp_path / 'aei1' / 'collect' / 'v0009-abc'
    write_group(coll, 'g000000', 'auth:x:1', [rollout('auth:x:1', 0, 'task_success', 'pass'),
                                              rollout('auth:x:1', 1, 'semantic_mismatch', 'fail', evidence='Maria')])
    write_group(coll, 'g000001', 'auth:held:1', [rollout('auth:held:1', 0, 'task_success', 'pass', evidence='H')])
    seen = {}

    def held_excluded_ids(manifest, tasks, held):
        seen['tasks'] = set(tasks)
        return {'auth:held:1'}
    monkeypatch.setattr(S, 'held_excluded_ids', held_excluded_ids)
    monkeypatch.setattr(S, 'held_closure', lambda manifest: ((set(), set(), set())))
    monkeypatch.setattr(cle.data_registry, 'role_path', lambda role: '/held/manifest.json')
    monkeypatch.setattr(cle.data_registry, 'require_registered', lambda paths: None)
    return coll, seen


def extract_args(coll, out, **kw):
    return SimpleNamespace(out=str(out), collect_root=None, collection=[str(coll)], authored_pool=[], work=None,
                           held_manifest=None, kinds=kw.get('kinds', 'decision,readout,entry'),
                           max_per_site=kw.get('max_per_site', 64), workers=1, limit_units=0, no_register=True)


def read_sites(out):
    return [json.loads(line) for line in S.zstd_lines(Path(out) / 'sites.jsonl.zst')]


def test_signature_and_answer_parsing():
    assert cle.return_type('() => RetimeBoardResult') == 'RetimeBoardResult'
    assert cle.return_type('(a: { x: (y: number) => string }) => Neuralese<{ f: string[] }>') == 'Neuralese<{ f: string[] }>'
    assert cle.return_type('verify(evidence: string): "pass" | "fail"') == '"pass" | "fail"'
    assert cle.opening_signature('You are inside this call: f(a: string): T\nmore') == ('f', 'f(a: string): T')
    msgs = decision_turn('c', None, 'e', 'pass')['messages']
    assert cle.decision_labels(msgs) == ['pass', 'fail', 'needs-review']
    assert cle.parse_answer('decision', '"fail"', [], ['pass', 'fail']) == ('decision_label', 'fail')
    assert cle.parse_answer('decision', '"maybe"', [], ['pass', 'fail'])[0] == 'invalid'
    assert cle.parse_answer('decision', '```json\n"pass"\n```', [], ['pass'])[0] == 'decision_label'
    assert cle.parse_answer('entry', '', [('return_result', {'status': 'success', 'value': 3})]) == ('return_value', 3)
    assert cle.parse_answer('entry', '', [('eval', {'code': 'x'})])[0] == 'action'
    text, calls = cle.normalize_reply('', [{'function': {'name': 'return_result', 'arguments': '{"value": 1}'}}])
    assert calls == [('return_result', {'value': 1})]


def test_extract_real_shape_merges_samples_and_excludes_held(collection, tmp_path):
    coll, seen = collection
    summary = cle.extract(extract_args(coll, tmp_path / 'out'))
    assert seen['tasks'] == {'auth:x:1', 'auth:held:1'}   # the exclusion saw every task row the sources name
    f = summary['funnel']
    assert f['episodes'] == 3 and f['episodes_held_excluded'] == 1
    assert f['calls:entry'] == 2 and f['calls:decision'] == 2 and f['calls:step'] == 2
    rows = read_sites(tmp_path / 'out')
    assert not any(r['task_id'] == 'auth:held:1' for r in rows)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r['kind'], []).append(r)
    assert 'step' not in by_kind   # not asked for
    # the root call's request is identical in both samples: one row, both observations, its exact prompt ids
    (entry,) = by_kind['entry']
    assert entry['n_observations'] == 2 and entry['role'] == 'root' and entry['definition_id'] == 'retime_board'
    assert entry['prompt_ids'] == [1, 2, 3] and entry['check']['returns'] == 'RetimeBoardResult'
    assert 'type RetimeBoardResult' in entry['check']['declarations']
    assert entry['messages'][1]['content'].startswith('You are inside this call: retime_board()')   # the real request
    assert entry['policy_success_any'] is True
    # the child decision differs by its argument: two rows, one site, labels, parent function, no token-id lists
    decisions = by_kind['decision']
    assert len(decisions) == 2 and len({d['site_id'] for d in decisions}) == 1
    assert {d['policy_answer']['key'] for d in decisions} == {cle.answer_key('decision_label', 'pass'),
                                                               cle.answer_key('decision_label', 'fail')}
    assert all(d['labels'] == ['pass', 'fail', 'needs-review'] and d['parent_definition'] == 'retime_board'
               for d in decisions)
    assert all('reasoning_sample_ids' not in d['decision_reasoning'] for d in decisions)
    failed = next(d for d in decisions if d['observations'][0]['outcome'] == 'semantic_mismatch')
    assert failed['policy_success_any'] is False


def test_extract_site_cap_and_steps(collection, tmp_path):
    coll, _ = collection
    summary = cle.extract(extract_args(coll, tmp_path / 'out', kinds='decision,step', max_per_site=1))
    assert summary['funnel']['site_cap:decision'] == 1 and summary['funnel']['admitted:step'] == 1


class FakeHelper:
    """Answers chat requests from a script and type checks through the real helper when node and ts-host are built."""
    replies = {}

    def __init__(self, *a, **k):
        import queue
        self.out = queue.Queue()

    def send(self, request):
        if request['op'] == 'check':
            ok = not isinstance(request['value'], str)
            self.out.put({'id': request['id'], 'check': {'checked': True, 'ok': ok}})
            return
        reply = self.replies[request['messages'][-1]['content'][-40:]].pop(0)
        self.out.put({'id': request['id'], 'turn': {'calls': reply.get('calls', []), 'text': reply.get('text', ''),
                                                       'reasoning': 'teacher thinks', 'prompt_tokens': 100,
                                                       'completion_tokens': 10}})

    def answers(self):
        while (item := self.out.get()) is not None:
            yield item

    def close(self):
        self.out.put(None)


def test_teach_agreement_type_check_and_resume(collection, tmp_path, monkeypatch):
    coll, _ = collection
    cle.extract(extract_args(coll, tmp_path / 'sites'))
    rows = read_sites(tmp_path / 'sites')
    entry = next(r for r in rows if r['kind'] == 'entry')
    decision_tail = rows[[r['kind'] for r in rows].index('decision')]['messages'][-1]['content'][-40:]
    FakeHelper.replies = {
        decision_tail: [{'text': '"fail"'}, {'text': '"fail"'}, {'text': '"pass"'},     # first decision: 2/3 agree
                        {'text': '"pass"'}, {'text': 'maybe'}, {'text': '"fail"'}],     # second: no majority of 2/3
        entry['messages'][-1]['content'][-40:]: [
            {'calls': [['return_result', {'status': 'success', 'value': 'wrong type'}]]}] * 3,
    }
    monkeypatch.setattr(cle, 'Helper', FakeHelper)
    spec = json.dumps({'model_id': 'fake-teacher', 'endpoint': 'http://127.0.0.1:9/v1'})
    args = SimpleNamespace(sites=str(tmp_path / 'sites'), out=str(tmp_path / 'teach'), teacher=['fake-teacher'],
                           teachers_file=str(tmp_path / 'none.json'), teacher_spec=[spec], kinds='decision,entry',
                           samples=3, min_agreement=2 / 3, temperature=1.0, max_tokens=100, limit=0, per_site=4,
                           max_prompt_chars=60000, seed=0, concurrency=1, min_interval_ms=0, max_requests=0,
                           max_total_tokens=0, max_errors=5, retries=2, usd_per_mtok_in=1.0, usd_per_mtok_out=2.0,
                           dry_run=False, no_register=True)
    (tmp_path / 'none.json').write_text('[]')
    summary = cle.teach(args)
    assert summary['funnel']['requests_sent'] == 9
    assert summary['stats']['verified'] == 1 and summary['stats']['verified:decision'] == 1
    out = [json.loads(line) for line in S.zstd_lines(tmp_path / 'teach' / 'call-level.jsonl.zst')]
    verified = [r for r in out if r['verified']]
    assert len(verified) == 1 and verified[0]['value'] == 'fail' and verified[0]['agreement'] == pytest.approx(2 / 3)
    assert verified[0]['teacher_reasoning'] == 'teacher thinks' and verified[0]['teacher_reasoning_weight'] == 0.0
    assert verified[0]['target'] == {'role': 'assistant', 'content': '"fail"'}   # no reasoning in the target
    typed = next(r for r in out if r['kind'] == 'entry')
    assert typed['basis'] == 'type_rejected' and not typed['verified']   # agreed, but the runtime's check rejects it
    assert summary['spend']['fake-teacher']['requests'] == 9 and summary['usd_estimate']['fake-teacher'] > 0
    # resume: every sample answered, nothing is sent again
    again = cle.teach(args)
    assert again['funnel']['requests_todo'] == 0 and again['stats'] == summary['stats']


def test_type_check_is_the_runtimes():
    """The helper's check is ts-host's coerce against the declared return type and the rendered declarations."""
    try:
        from portable_paths import node_executable
        node = node_executable()
    except RuntimeError:
        pytest.skip('node unavailable')
    if not (ROOT / 'ts-host/dist/native/values.js').exists():
        pytest.skip('ts-host is not built')
    helper = cle.Helper(1, 0, node=node)
    check = {'returns': 'Neuralese<Board>', 'declarations': '/** b */\ntype Board = { flights: { flight: string }[] };'}
    verdicts = cle.check_values(helper, [('ok', {'flights': [{'flight': 'NW 1'}]}, check),
                                         ('bad', {'flights': 'x'}, check),
                                         ('label', 'maybe', {'returns': '"pass" | "fail"', 'declarations': ''})])
    helper.close()
    assert verdicts['ok'] == {'checked': True, 'ok': True}
    assert verdicts['bad']['checked'] and not verdicts['bad']['ok']
    assert not verdicts['label']['ok']
