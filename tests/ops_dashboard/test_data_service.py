"""Targeted tests of the ops page's data service: line indexes, normalizers, token masks, edit layers, catalog constants."""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'ops_dashboard'))
import data_service  # noqa: E402
import data_items  # noqa: E402
import data_edits  # noqa: E402
import data_catalog  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(data_items, 'CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(data_items, '_IDX', {})
    monkeypatch.setattr(data_items, '_LINE_CACHE', {})
    monkeypatch.setattr(data_edits, 'LAYER', str(tmp_path / 'data-edits'))
    monkeypatch.setattr(data_edits, 'ERRATA_FILE', str(tmp_path / 'pool-errata' / 'ops-dashboard.jsonl'))
    monkeypatch.setattr(data_edits, 'INBOX', str(tmp_path / 'inbox.jsonl'))
    monkeypatch.setattr(data_edits, 'is_held', lambda tid, row=None: tid.startswith('held:'))


def write_jsonl(path, recs, zst=False):
    text = ''.join(json.dumps(r) + '\n' for r in recs).encode()
    if zst:
        import zstandard
        text = zstandard.ZstdCompressor().compress(text)
    path.write_bytes(text)
    return str(path)


def test_jsonl_since_offsets(tmp_path):
    p = tmp_path / 'm.jsonl'
    write_jsonl(p, [{'ts': 1, 'n': i} for i in range(5)])
    recs, off = data_service.jsonl_since(str(p), 0)
    assert [r['n'] for r in recs] == [0, 1, 2, 3, 4] and off == p.stat().st_size
    with open(p, 'a') as f:
        f.write(json.dumps({'ts': 2, 'n': 5}) + '\n{"partial":')
    recs, off2 = data_service.jsonl_since(str(p), off)
    assert [r['n'] for r in recs] == [5] and off2 < p.stat().st_size      # the partial line waits for its newline
    tail, _ = data_service.jsonl_since(str(p), 0, max_bytes=40)
    assert tail and tail[-1]['n'] == 5 and all('n' in r for r in tail)    # a cut first line is skipped, not parsed


@pytest.mark.parametrize('zst', [False, True])
def test_line_index_read_and_grep(tmp_path, zst):
    rows = [{'id': f't{i}', 'family': 'fa' if i % 2 else 'fb', 'semantics': {'files': {'a.nl': 'x'}}} for i in range(6)]
    p = write_jsonl(tmp_path / ('c.jsonl.zst' if zst else 'c.jsonl'), rows, zst)
    idx = data_items.line_index(p, data_items.task_summary)
    assert [s['id'] for _, s in idx] == [r['id'] for r in rows]
    loc = idx[3][0]
    assert data_items.read_at(p, loc)['id'] == 't3'
    assert set(data_items.grep_locs(p, 'T5')) == {idx[5][0]}
    page = data_items.jsonl_items(p, data_items.task_summary, {'filter': {'family': 'fa'}, 'limit': 2})
    assert page['total'] == 3 and len(page['rows']) == 2 and dict(page['facets']['family'])['fb'] == 3


def test_turns_keep_only_new_context_and_name_calls():
    sys_m, user = {'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'U'}
    t0 = {'messages': [sys_m, user], 'invocation_id': 'inv', 'target': {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c1', 'function': {'name': 'eval', 'arguments': '{"code":"1+1","finish":true}'}}]}, 'function': 'unknown'}
    t1 = {'messages': [sys_m, user, {'role': 'assistant', 'content': '', 'tool_calls': t0['target']['tool_calls']},
                       {'role': 'tool', 'tool_call_id': 'c1', 'content': '2'}], 'invocation_id': 'inv',
          'target': {'role': 'assistant', 'content': 'done', 'reasoning_content': 'think'}, 'function': 'unknown'}
    e = {'turns': [t0, t1], 'execution_graph': [{'call_id': 'inv', 'definition': {'id': 'answer'}, 'phase': 'start'}],
         'group': 3, 'sample': 1, 'reward': {'quality': 1.0}}
    out = data_items.normalize_rollout(e, advantage=1.2)
    c0, c1 = out['calls']
    assert [m['role'] for m in c0['context']] == ['system', 'user'] and c0['target']['tool_calls'][0]['parsed']['code'] == '1+1'
    assert c1['context_shared'] == 3 and [m['role'] for m in c1['context']] == ['tool']    # only the tool result is new
    assert c1['target']['reasoning'] == 'think' and c0['function'] == 'answer'


def test_group_advantages_match_grpo():
    assert data_items.group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]
    adv = data_items.group_advantages([1, 0, 0, 0])
    assert adv[0] > 0 and all(a < 0 for a in adv[1:]) and abs(sum(adv)) < 1e-3


def test_sampled_token_masks(monkeypatch):
    ids = {'</think>': 9, '<tool_call>': 7, '<|im_end|>': 8}
    monkeypatch.setattr(data_items, 'token_pieces', lambda xs: [str(x) for x in xs])
    monkeypatch.setattr(data_items, 'special_id', lambda t: ids.get(t))
    turn = {'sampling': {'samples': [{'token_id': t, 'logp': lp, 'input_role': 'ordinary'} for t, lp in
                                     [(1, -0.1), (9, -0.2), (2, -9.0), (9, -0.3), (3, -0.1), (7, -0.1)]]}}
    pos = data_items.sampled_tokens(turn, advantage=1.0)
    assert pos['tail_masked'] == [False, False, True, False, False, False]
    assert pos['second_close_masked'] == [False, False, False, True, True, False]   # second close up to the tool call
    neg = data_items.sampled_tokens(turn, advantage=-1.0)
    assert not any(neg['tail_masked']) and not any(neg['second_close_masked'])


def edit(op, action, kind='task', tid='auth:x:1', ds='pool:p', **kw):
    return {'operationId': op, 'author': 'dev', 'action': action, 'text': 'why',
            'target': {'kind': kind, 'dataset': ds, 'id': tid, 'label': 'fam_x'}, **kw}


def test_edit_layers_exclude_include_idempotent(tmp_path):
    r = data_edits.edits_apply(edit('op1', 'exclude'))
    assert r['status'] == 'applied' and 'pool-errata' in r['files']
    errata = [json.loads(l) for l in open(data_edits.ERRATA_FILE)]
    assert errata == [{'id': 'auth:x:1', 'family': 'fam_x', 'reason': 'why', 'source': 'ops-dashboard edit op1', 'type': 'excluded-by-review'}]
    assert data_edits.edits_apply(edit('op1', 'exclude')) == r                       # a retried application is a no-op
    assert len(open(data_edits.LAYER + '/log.jsonl').readlines()) == 1
    data_edits.edits_apply(edit('op2', 'include'))
    assert open(data_edits.ERRATA_FILE).read() == ''
    assert not list(Path(data_edits.ERRATA_FILE).parent.glob('*.tmp-write'))


def test_edit_layers_refuse_held_and_route_kinds(tmp_path):
    r = data_edits.edits_apply(edit('h1', 'exclude', tid='held:t'))
    assert r['status'] == 'refused' and not Path(data_edits.ERRATA_FILE).exists()
    data_edits.edits_apply(edit('s1', 'exclude', kind='sft_row', tid='row1', ds='sft:mix'))
    excl = [json.loads(l) for l in open(data_edits.LAYER + '/exclusions.jsonl')]
    assert [x['id'] for x in excl] == ['row1'] and open(data_edits.ERRATA_FILE).read() == ''   # SFT rows never become pool errata
    data_edits.edits_apply(edit('c1', 'correct', correction=['{"a":', '1}'], correctionFormat='json', field='oracle.expected'))
    corr = [json.loads(l) for l in open(data_edits.LAYER + '/corrections.jsonl')]
    assert corr[0]['field'] == 'oracle.expected' and data_edits._log()[-1]['correction'] == '{"a":1}'
    state = data_edits.edits_layers(None)['state']
    assert state['sft:mix|row1']['excluded']['op'] == 's1' and state['pool:p|auth:x:1']['corrections'][0]['op'] == 'c1'


def test_inbox_sync(tmp_path):
    v = {'asks': [{'operationId': 'a1', 'kind': 'ask', 'subject': {'kind': 'unit', 'id': 'u'}, 'text': 'q', 'author': 'b'}],
         'answers': [{'ask': 'a1', 'text': 'yes', 'author': 'n', 'operationId': 'x'}]}
    assert data_edits.inbox_sync(v)['changed'] and not data_edits.inbox_sync(v)['changed']
    line = json.loads(open(data_edits.INBOX).read())
    assert line['answers'][0]['text'] == 'yes'



def test_inbox_sync_accepts_chat_style_messages(tmp_path):
    """The same inbox.jsonl from chat-style agent messages: no inReplyTo = an ask (data carries kind and subject)."""
    old = {'asks': [{'operationId': 'a1', 'kind': 'ask', 'subject': {'kind': 'unit', 'id': 'u'}, 'text': 'q', 'author': 'b'}],
           'answers': [{'ask': 'a1', 'text': 'yes', 'author': 'n', 'operationId': 'x'}]}
    data_edits.inbox_sync(old)
    want = open(data_edits.INBOX).read()
    chat = {'messages': [{'author': 'b', 'operationId': 'a1', 'text': 'q', 'data': {'kind': 'ask', 'subject': {'kind': 'unit', 'id': 'u'}}},
                         {'author': 'n', 'operationId': 'x', 'text': 'yes', 'inReplyTo': 'a1'}]}
    assert data_edits.inbox_sync(chat) == {'asks': 1, 'changed': False} and open(data_edits.INBOX).read() == want


def test_code_constants_parse(tmp_path, monkeypatch):
    src = tmp_path / 'grpo.py'
    src.write_text("POOLS = [resolve_str('data_hdd', 'a', 'cases.jsonl'),\n  resolve_str('data_nvme', 'b', 'cases.jsonl')]\n"
                   "HELD = resolve_str('data_nvme', 'x', 'manifest.json')\nOTHER = 1\n")
    monkeypatch.setattr(data_catalog, 'GRPO', str(src))
    monkeypatch.setattr(data_catalog, '_CODE', {})
    c = data_catalog.code_constants()
    hdd, nvme = data_catalog.ROOTS['data_hdd'], data_catalog.ROOTS['data_nvme']
    assert c['POOLS']['value'] == [f'{hdd}/a/cases.jsonl', f'{nvme}/b/cases.jsonl'] and c['POOLS']['line'] == 1
    assert c['HELD']['value'] == f'{nvme}/x/manifest.json' and 'OTHER' not in c


def test_live_grpo_constants_resolve():
    """The catalog reads the trainer's real default pools (grpo.py POOLS, or its registry role): each resolves to a path."""
    data_catalog._CODE.clear()
    c = data_catalog.code_constants()
    # registry pools are JSONL task files (cases.jsonl, and since the M2 migration also tasks.jsonl)
    assert len(c['POOLS']['value']) >= 2 and all(p.endswith('.jsonl') for p in c['POOLS']['value'])


def test_tick_carries_the_agents_inbox_listener_heartbeat(tmp_path, monkeypatch):
    """`inbox.sh wait` touches <dashboard>/inbox.listener; the tick (pushed to the page every 10 s) carries its mtime, null when never started."""
    monkeypatch.setattr(data_service, 'DASH', str(tmp_path))
    monkeypatch.setattr(data_service, 'OUT', str(tmp_path / 'out'))
    assert data_service.answer('ops.tick', None)['listener'] is None
    (tmp_path / 'inbox.listener').write_text('')
    import os
    os.utime(tmp_path / 'inbox.listener', (1_700_000_000, 1_700_000_000))
    assert data_service.answer('ops.tick', None)['listener'] == 1_700_000_000_000
