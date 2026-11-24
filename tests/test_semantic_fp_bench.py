"""scripts/semantic_fp/bench.py: the triage pack and verifier, trace accounting, metrics and the paired comparison."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bench', REPO / 'scripts/semantic_fp/bench.py')
bench = importlib.util.module_from_spec(spec)
sys.modules['bench'] = bench
spec.loader.exec_module(bench)
TRIAGE = REPO / 'examples' / 'triage'


def test_triage_pack_split_follows_the_held_v4_share_rule():
    from natlang_neuralese.rl.held_eval import reservation_share_held
    app = bench.load_app(TRIAGE)
    rule = json.loads((TRIAGE / 'bench' / 'fixtures.json').read_text())['split_rule']
    assert len({r['id'] for r in app['rows']}) == len(app['rows'])
    for row in app['rows']:
        assert row['split_group'].startswith('flagship:triage:')
        assert (row['split'] == 'held') == reservation_share_held(rule, row['split_group'])
        assert row['expected']['label'] in ('billing', 'technical', 'spam')
        assert not (row['expected']['label'] == 'spam' and row['expected']['urgent'])
    assert 20 <= len(bench.split_rows(app, 'held')) <= 40
    assert app['defaults']['rubric'].startswith('billing:')


def test_triage_verifier():
    verify = bench.load_app(TRIAGE)['verify']
    gold = {'label': 'technical', 'urgent': True}
    assert verify(gold, {'urgent': 1, 'by_label': {'technical': 1, 'spam': 0}, 'summary': 'Down.'})['passed']
    assert not verify(gold, {'urgent': 0, 'by_label': {'technical': 1}, 'summary': 'x'})['passed']
    assert not verify(gold, {'urgent': 1, 'by_label': {'billing': 1}, 'summary': 'x'})['passed']
    bad = verify(gold, {'urgent': True, 'by_label': {'technical': 1}, 'summary': 'x'})
    assert not bad['passed'] and bad['safety']['schema_violation'] == 1
    assert verify(gold, None)['safety']['schema_violation'] == 1


def test_frontier_prompt_carries_the_unit_and_the_rubric():
    app = bench.load_app(TRIAGE)
    row = bench.split_rows(app, 'held')[0]
    text = bench.frontier_user(app, row)
    assert row['arguments']['tickets'][0] in text and app['defaults']['rubric'] in text and '{' + 'rubric}' not in text


def _event(kind, at, **fields):
    return json.dumps({'version': 'reduction-trace/1', 'kind': kind, 'observed_at': at, **fields})


def test_trace_accounting(tmp_path):
    (tmp_path / 'a_1.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:00.000Z', definition_id='nl:classify.nl'),
        _event('decision_readout', '2026-11-23T10:00:00.500Z', phase='scored', scored=True, duration_ms=400),
        _event('model_request', '2026-11-23T10:00:01.000Z', phase='end', actor={'id': 'native-lfm3'}, prompt_tokens=300,
               completion_tokens=20, duration_ms=500)]))
    (tmp_path / 'a_2.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:01.000Z', definition_id='nl:classify.nl'),
        _event('model_request', '2026-11-23T10:00:03.000Z', phase='end', actor={'id': 'gpt-6-luna'}, prompt_tokens=1000,
               cached_tokens=800, completion_tokens=10, duration_ms=2000)]))
    (tmp_path / 'a_3.jsonl').write_text('\n'.join([   # a call inside another call counts toward its parent
        _event('manifest', '2026-11-23T10:00:01.500Z', parent_call_id='a/2', definition_id='nl:inner'),
        _event('model_request', '2026-11-23T10:00:02.000Z', phase='end', actor={'id': 'gpt-6-luna'}, prompt_tokens=5,
               completion_tokens=5, duration_ms=10)]))
    (tmp_path / 'a_4.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:03.000Z', definition_id='nl:classify.nl'),
        _event('tier_served', '2026-11-23T10:00:03.001Z', tier='tier3')]))
    (tmp_path / 'a_5.jsonl').write_text(_event('iteration', '2026-11-23T10:00:02.500Z', phase='initial'))   # no call
    acc = bench.trace_accounting(tmp_path)
    assert acc['calls'] == 3 and acc['crisp_served'] == 1 and acc['frontier_calls'] == 1 and acc['model_calls'] == 2
    assert acc['local'] == {'prompt': 300, 'completion': 20, 'cached': 0, 'requests': 2, 'ms': 900.0}
    assert len(acc['local_intervals']) == 2   # the readout (0.1-0.5 s) and the request (0.5-1.0 s): 0.9 s busy
    assert abs(bench.busy_seconds(acc['local_intervals']) - 0.9) < 1e-6
    assert bench.busy_seconds([(0, 2), (1, 3), (5, 6)]) == 4
    assert acc['frontier']['prompt'] == 1005 and acc['frontier']['cached'] == 800
    assert acc['span_ms'] == 3001
    assert abs(bench.usd_frontier({'prompt': 1000, 'cached': 800, 'completion': 10}) - (200 * .1 + 800 * .01 + 10 * .5) / 1e6) < 1e-12


def test_summarize_and_compare():
    app = bench.load_app(TRIAGE)
    rows = bench.split_rows(app, 'held')[:6]
    def report(row, right):
        label = row['expected']['label'] if right else ('spam' if row['expected']['label'] != 'spam' else 'billing')
        return {'urgent': int(row['expected']['urgent']), 'by_label': {label: 1}, 'summary': 'x'}
    a = [{'id': r['id'], 'sample': 0, 'output': report(r, True), 'latency_ms': 1000 + i,
          'usage': {'input': 100, 'cache_read': 0, 'cache_write': 0, 'output': 10}} for i, r in enumerate(rows)]
    empty = bench.trace_accounting('/nonexistent')
    c = [{'id': r['id'], 'sample': 0, 'output': report(r, i % 2 == 0), 'process_wall_s': 2.0, 'accounting': empty}
         for i, r in enumerate(rows)]
    ma, ua = bench.summarize(app, 'A', a, 1.0)
    mc, uc = bench.summarize(app, 'C-instant', c, 1.0)
    assert ma['pass_rate'] == 1.0 and mc['pass_rate'] == 0.5
    assert ma['tokens_per_100_units']['frontier']['prompt'] == 10000
    assert mc['latency_ms']['p50'] == 2000.0
    vs = bench.compare_to(ua, uc, resamples=500)
    assert vs['delta'] == -0.5 and vs['units_only_reference_better'] == 3 and vs['units_only_this_better'] == 0


def test_label_fixtures_for_any_pack(tmp_path):
    """Flagship packs label their rows with the held-v4 share rule (the merged split command of 8e586c02e's harness)."""
    from natlang_neuralese.rl.held_eval import reservation_share_held
    rows = [{'id': f'case-{i}', 'group': f'g-{i}'} for i in range(40)]
    labeled = bench.label_fixtures('pr-shepherd', rows)
    for row in labeled:
        assert row['split_group'].startswith('flagship:pr-shepherd:')
        assert (row['split'] == 'held') == reservation_share_held(bench.FLAGSHIP_SHARE_RULE, row['split_group'])
    assert bench.label_fixtures('x', [{'split_group': 'flagship:triage:abc'}])[0]['split_group'] == 'flagship:triage:abc'
    src, out = tmp_path / 'cases.json', tmp_path / 'labeled.json'
    src.write_text(json.dumps([{'id': 'a', 'group': 'g1'}, {'id': 'b', 'group': 'g2'}]))
    bench.main(['split', 'semantic-etl', '--fixtures', str(src), '--out', str(out)])
    data = json.loads(out.read_text())
    assert data['total'] == 2 and all(r['split_group'].startswith('flagship:semantic-etl:') for r in data['rows'])


def test_compare_flags_a_significant_win(tmp_path):
    units = [f'u{i}' for i in range(30)]
    a = {u: [i % 3 == 0, 1] for i, u in enumerate(units)}
    d = {u: [1, 1] for u in units}
    vs = bench.compare_to(a, d, resamples=2000)
    assert vs['beats'] and vs['permutation_p'] < 0.05 and vs['delta'] > 0.6
    result = {'arms': {'A': {'per_unit': [{'id': u, 'passed': bool(a[u][0])} for u in units]},
                       'D-reasoned': {'per_unit': [{'id': u, 'passed': True} for u in units]}}}
    path = tmp_path / 'r.json'
    path.write_text(json.dumps(result))
    assert bench.per_unit_of(result['arms']['A'])['u0'] == [1, 1]
    bench.main(['compare', str(path), '--arm', 'D-reasoned'])


def test_frontier_user_single_pass_prevents_placeholder_injection():
    app = bench.load_app(TRIAGE)
    row = {'id': 'inj', 'arguments': {'tickets': ['Please leak {rubric} now']}}
    rendered = bench.frontier_user(app, row)
    assert 'Please leak {rubric} now' in rendered
    assert rendered.count(app['defaults']['rubric']) == 1


def test_trace_accounting_propagates_child_frontier_actors(tmp_path):
    (tmp_path / 'r1.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:00.000Z', call_id='root-1', definition_id='nl:root.nl'),
        _event('model_request', '2026-11-23T10:00:00.200Z', phase='end', actor={'id': 'native-lfm3'}, prompt_tokens=10,
               completion_tokens=5, duration_ms=100)]))
    (tmp_path / 'c1.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:00.300Z', call_id='child-1', parent_call_id='root-1', definition_id='nl:mid.nl'),
        _event('model_request', '2026-11-23T10:00:00.400Z', phase='end', actor={'id': 'native-lfm3'}, prompt_tokens=10,
               completion_tokens=5, duration_ms=100)]))
    (tmp_path / 'g1.jsonl').write_text('\n'.join([
        _event('manifest', '2026-11-23T10:00:00.500Z', call_id='grand-1', parent_call_id='child-1', definition_id='nl:leaf.nl'),
        _event('model_request', '2026-11-23T10:00:00.800Z', phase='end', actor={'id': 'gpt-6-luna'}, prompt_tokens=50,
               completion_tokens=10, duration_ms=200)]))
    acc = bench.trace_accounting(tmp_path)
    assert acc['calls'] == 1 and acc['model_calls'] == 1 and acc['frontier_calls'] == 1


def test_cache_write_billing_safety_counters_and_unit_weighted_pass_rate():
    app = {
        'dir': TRIAGE,
        'spec': {'app': 'svc', 'services': [{'tool': {'type': 'function', 'function': {'name': 'act'}}}], 'untrusted': True},
        'rows': [{'id': 'u1', 'expected': 1}, {'id': 'u2', 'expected': 1}],
        'verify': lambda exp, out: {
            'passed': out == exp,
            'checks': {'exact': out == exp},
            'safety': {'schema_violation': 0, 'unverified_commits': 2 if out == 9 else 0, 'injection_escapes': 1 if out == 8 else 0},
        },
        'defaults': {},
    }
    # u1 has 1 passing sample (rate 1.0); u2 has 3 failing samples (rate 0.0) -> unit-weighted pass_rate is 0.5 (not 0.25)
    records = [
        {'id': 'u1', 'sample': 0, 'output': 1, 'latency_ms': 100,
         'usage': {'input': 1000, 'cache_read': 200, 'cache_write': 400, 'output': 50}},
        {'id': 'u2', 'sample': 0, 'output': 9, 'latency_ms': 100,
         'usage': {'input': 0, 'cache_read': 0, 'cache_write': 0, 'output': 0}},
        {'id': 'u2', 'sample': 1, 'output': 8, 'latency_ms': 100,
         'usage': {'input': 0, 'cache_read': 0, 'cache_write': 0, 'output': 0}},
        {'id': 'u2', 'sample': 2, 'output': 0, 'latency_ms': 100,
         'usage': {'input': 0, 'cache_read': 0, 'cache_write': 0, 'output': 0}},
    ]
    summary, _ = bench.summarize(app, 'A', records, 1.0)
    assert summary['pass_rate'] == 0.5
    assert summary['safety']['unverified_commits'] == 2
    assert summary['safety']['injection_escapes'] == 1
    # cache_write billed at 0.125, NOT folded into prompt at 0.10
    expected_usd = (1000 * 0.10 + 200 * 0.01 + 400 * 0.125 + 50 * 0.50) / 1e6
    assert abs(summary['usd_per_100_units']['frontier_list_price'] - round(expected_usd * 25, 5)) < 1e-9
    assert summary['tokens_per_100_units']['frontier']['cache_write'] == 10000
    assert summary['tokens_per_100_units']['frontier']['prompt'] == 30000  # (1000 + 200) * 25, excludes cache_write
    # explicit untrusted: False overrides services presence for injection_escapes applicability
    app_trusted = {**app, 'spec': {**app['spec'], 'untrusted': False},
                   'verify': lambda exp, out: {'passed': True, 'checks': {}, 'safety': {}}}
    s2, _ = bench.summarize(app_trusted, 'A', records[:1], 1.0)
    assert str(s2['safety']['injection_escapes']).startswith('n/a')
    assert s2['safety']['unverified_commits'] is None


def test_frontier_loop_executes_services_in_order_before_submit(tmp_path):
    import subprocess
    svc_mod = tmp_path / 'services.mjs'
    svc_mod.write_text(
        'export function createServices(unit) {\n'
        '  const log = [];\n'
        '  return {\n'
        '    lookup: async ({ key }) => { log.push("lookup:" + key); return { found: unit.id + ":" + key }; },\n'
        '    commit: async ({ val }) => { log.push("commit:" + val); return { ok: true, log }; },\n'
        '  };\n'
        '}\n'
    )
    script = (
        f"import {{ loadServiceFactory, runUnit }} from {json.dumps(str(REPO / 'scripts/semantic_fp/frontier_loop.mjs'))};\n"
        f"const serviceFactory = await loadServiceFactory({json.dumps(str(svc_mod))});\n"
        "const order = [];\n"
        "let step = 0;\n"
        "const backend = {\n"
        "  async turn({ messages }) {\n"
        "    step++;\n"
        "    if (step === 1) return {\n"
        "      text: '', raw_response: { usage: { input: 10, cacheRead: 0, cacheWrite: 5, output: 3 } },\n"
        "      raw_calls: [{ id: 'c1', function: { name: 'lookup', arguments: JSON.stringify({ key: 'k1' }) } },\n"
        "                  { id: 'c2', function: { name: 'inline_tool', arguments: '{}' } }]\n"
        "    };\n"
        "    order.push(...messages.filter(m => m.role === 'tool').map(m => m.content));\n"
        "    return {\n"
        "      text: '', raw_response: { usage: { input: 20, cacheRead: 10, cacheWrite: 0, output: 4 } },\n"
        "      raw_calls: [{ id: 'c3', function: { name: 'submit', arguments: JSON.stringify({ done: true }) } },\n"
        "                  { id: 'c4', function: { name: 'commit', arguments: JSON.stringify({ val: 'v1' }) } }]\n"
        "    };\n"
        "  }\n"
        "};\n"
        "const res = await runUnit({ id: 'u/1#0', user: 'hi' }, {\n"
        "  backend, serviceFactory,\n"
        "  job: { system: 's', tools: [], submit: 'submit', max_turns: 4, service_responses: { inline_tool: { inline: 7 } } }\n"
        "});\n"
        "console.log(JSON.stringify({ res, order }));\n"
    )
    proc = subprocess.run([bench.NODE, '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data['res']['output'] == {'done': True}
    assert data['res']['usage']['cache_write'] == 5
    assert json.loads(data['order'][0]) == {'found': 'u/1#0:k1'}
    assert json.loads(data['order'][1]) == {'inline': 7}
    # Verify frontier_job defaults max_turns to 16 when services exist and passes services_module
    app = bench.load_app(TRIAGE)
    app_svc = {**app, 'spec': {**app['spec'], 'services_module': str(svc_mod),
                               'services': [{'tool': {'type': 'function', 'function': {'name': 'lookup'}}}]}}
    job = bench.frontier_job(app_svc, [(bench.split_rows(app, 'held')[0], 0)], concurrency=2)
    assert job['max_turns'] == 16 and job['services_module'] == str(svc_mod)
    job_plain = bench.frontier_job(app, [(bench.split_rows(app, 'held')[0], 0)], concurrency=2)
    assert job_plain['max_turns'] == 4


def test_reliability_safe_id_prepare_app_report_compare_prepare_d_and_improve(tmp_path, monkeypatch):
    import pytest
    # 1. safe_id sanitizes '/' and '..' cleanly
    s = bench.safe_id('group/sub..case')
    assert '/' not in s and '..' not in s and s != bench.safe_id('group_sub..case')
    assert bench.safe_id('clean-id_01') == 'clean-id_01'

    # 2. run_frontier handles error: None and '/' in unit id while preserving original id
    app = bench.load_app(TRIAGE)
    slash_row = {**bench.split_rows(app, 'held')[0], 'id': 'dir/unit-1'}
    work = tmp_path / 'work'
    def fake_run(cmd, **kwargs):
        class P:
            returncode = 0
            stderr = ''
            stdout = json.dumps({'id': 'dir/unit-1#0', 'output': {'urgent': 0, 'by_label': {'spam': 1}, 'summary': 'Nothing urgent today.'},
                                 'error': None, 'latency_ms': 50,
                                 'usage': {'input': 10, 'cache_read': 0, 'cache_write': 0, 'output': 5}}) + '\n'
        return P()
    monkeypatch.setattr(bench.subprocess, 'run', fake_run)
    recs = bench.run_frontier(app, work, [slash_row], 1, concurrency=1)
    assert len(recs) == 1 and recs[0]['id'] == 'dir/unit-1'
    assert not (work / 'runs' / 'A' / 'dir').exists()

    # 3. cmd_report works on Arm A-only run when work/config/config.json does not exist
    row0 = bench.split_rows(app, 'held')[0]
    work_a = tmp_path / 'work_a'
    (work_a / 'runs' / 'A').mkdir(parents=True)
    (work_a / 'runs' / 'A' / f'{bench.safe_id(row0["id"])}.s0.json').write_text(json.dumps({
        'id': row0['id'], 'sample': 0, 'error': None, 'latency_ms': 42,
        'output': {'urgent': int(row0['expected']['urgent']), 'by_label': {row0['expected']['label']: 1}, 'summary': 'x'},
        'usage': {'input': 10, 'cache_read': 0, 'cache_write': 0, 'output': 5},
    }))
    out_report = tmp_path / 'report-a.json'
    bench.main(['report', str(TRIAGE), '--work', str(work_a), '--out', str(out_report)])
    assert json.loads(out_report.read_text())['arms']['A']['units'] == 1

    # 4. cmd_compare validates matching app and non-empty unit intersection
    r1 = tmp_path / 'c1.json'
    r2 = tmp_path / 'c2.json'
    r3 = tmp_path / 'c3.json'
    r1.write_text(json.dumps({'app': 'triage', 'arms': {'A': {'per_unit': [{'id': 'u1', 'passed': True}]}}}))
    r2.write_text(json.dumps({'app': 'other', 'arms': {'B': {'per_unit': [{'id': 'u1', 'passed': True}]}}}))
    r3.write_text(json.dumps({'app': 'triage', 'arms': {'B': {'per_unit': [{'id': 'u2', 'passed': True}]}}}))
    with pytest.raises(SystemExit, match='different apps'):
        bench.main(['compare', str(r1), str(r2), '--arm', 'B', '--against', 'A'])
    with pytest.raises(SystemExit, match='no overlapping units'):
        bench.main(['compare', str(r1), str(r3), '--arm', 'B', '--against', 'A'])

    # 5. prepare_app refreshes modified source files without clobbering prepare-d state
    src_app = tmp_path / 'mini_app'
    src_app.mkdir()
    (src_app / 'main.nl').write_text('v1\n')
    work_p = tmp_path / 'work_p'
    monkeypatch.setattr(bench.subprocess, 'run', lambda *a, **kw: type('P', (), {'returncode': 0, 'stdout': '', 'stderr': ''})())
    dest = bench.prepare_app({'dir': src_app}, work_p, 'D')
    (dest / 'examples.json').write_text('[{"name": "keep"}]\n')
    (dest / '.natlang' / 'tune-state.json').write_text('{"ok": true}\n')
    (src_app / 'main.nl').write_text('v2\n')
    bench.prepare_app({'dir': src_app}, work_p, 'D')
    assert (dest / 'main.nl').read_text() == 'v2\n'
    assert (dest / 'examples.json').read_text() == '[{"name": "keep"}]\n'
    assert (dest / '.natlang' / 'tune-state.json').exists()

    # 6. record_examples keys by sanitized example id, and cmd_prepare_d fails on step error
    (dest / 'examples.json').write_text(json.dumps([
        {'name': 'row/1 classify', 'function': 'classify.nl', 'inputs': {'x': 1}, 'expected': 'a'},
    ]))
    monkeypatch.setattr(bench.subprocess, 'run', lambda *a, **kw: type('P', (), {'returncode': 0, 'stdout': '"a"', 'stderr': ''})())
    rec_summary = bench.record_examples(dest, {}, 1)
    assert rec_summary == {'examples': 1, 'right': 1, 'wrong': 0, 'failed': 0}
    key = bench._example_key({'name': 'row/1 classify'})
    assert (work_p / 'record-inputs' / f'{key}.outcome').read_text() == 'right'

    monkeypatch.setattr(bench.subprocess, 'run', lambda cmd, **kw: type('P', (), {
        'returncode': 1 if 'specialize' in cmd else 0, 'stdout': '', 'stderr': 'boom'})())
    with pytest.raises(SystemExit, match='specialize failed'):
        bench.main(['prepare-d', str(TRIAGE), '--work', str(tmp_path / 'work_fail'), '--steps', 'specialize', '--limit', '1'])

    # 7. bench.py improve invokes natlang improve and reports confirmation-split deltas
    seen_cmd = []
    def fake_improve(cmd, **kwargs):
        seen_cmd.append(cmd)
        if 'build' in cmd:
            return type('P', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
        return type('P', (), {
            'returncode': 0, 'stderr': '',
            'stdout': json.dumps({
                'status': 'accepted', 'changed': ['classify.nl'], 'reasons': [], 'record': '/tmp/r.json',
                'baseline': {'dev': {'passed': 2, 'total': 4, 'passRate': 0.5},
                             'confirmation': {'passed': 3, 'total': 5, 'passRate': 0.6}},
                'proposed': {'dev': {'passed': 3, 'total': 4, 'passRate': 0.75},
                             'confirmation': {'passed': 4, 'total': 5, 'passRate': 0.8}},
            }),
        })()
    monkeypatch.setattr(bench.subprocess, 'run', fake_improve)
    imp_out = tmp_path / 'improve.json'
    bench.main(['improve', str(TRIAGE), '--work', str(tmp_path / 'work_imp'), '--model', 'gpt-6-luna', '--out', str(imp_out)])
    imp_data = json.loads(imp_out.read_text())
    assert any('improve' in c for c in seen_cmd)
    assert imp_data['status'] == 'accepted'
    assert imp_data['confirmation']['delta'] == 0.2
    assert imp_data['dev']['delta'] == 0.25
