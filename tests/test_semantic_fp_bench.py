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
    assert acc['local'] == {'prompt': 300, 'completion': 20, 'cached': 0, 'requests': 1, 'ms': 900.0}
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
