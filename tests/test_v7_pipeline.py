import json
import random

from natlang.corpus import index_pairs, split_programs
from natlang.gen import synth as S
from scripts.generate import run_program
from scripts.paraphrase_steps import valid_variant
from scripts.teacher_leaves import CHECKS
from natlang.gen.architectures import reconciliation


def test_style_override_preserves_program_randomness(monkeypatch):
    states = []
    for style in S.MARK_STYLES:
        monkeypatch.setenv('NATLANG_MARK_STYLE', style)
        rng = random.Random(7)
        assert S.mark_style(rng) == style
        states.append(rng.getstate())
    assert states[0] == states[1] == states[2]


def test_recovery_history_is_not_a_bad_supervised_target():
    prog = S.composed(random.Random(2))
    samples, _ = run_program(prog, recovery_rate=1)
    repaired = [s for s in samples if s.get('recovery')]
    assert repaired
    assert any(any(m['role'] == 'tool' and m['content'].startswith('error:') for m in s['messages']) for s in repaired)
    for s in samples:
        assert '__missing_value' not in s['native_target'] and "JSON.parse('{')" not in s['native_target']


def test_wrong_call_destination_is_history_followed_by_a_correct_local_call():
    samples, _ = run_program(reconciliation(random.Random(19)), recovery_rate=1)
    corrections = [s for s in samples if s.get('recovery') and 'join_events' in s['native_target']]
    assert corrections
    first = corrections[0]
    target = first['target']['tool_calls'][-1]['function']
    assert json.loads(target['arguments'])['to'] == 'let/joined'
    assert any(m['role'] == 'tool' and 'type-does-not-fit-slot' in m['content'] for m in first['messages'])
    bad = [m for m in first['messages'] if m.get('tool_calls')][-1]['tool_calls'][0]['function']
    assert json.loads(bad['arguments'])['to'] == 'return'


def test_phrase_filter_preserves_placeholders_and_rejects_listing_markers():
    base = 'return {field}'
    for candidate in ['- return {field}', '- [ ] return {field}', '=> {field}', '1. return {field}', 'return {other}']:
        assert not valid_variant(candidate, base)
    assert valid_variant('return ({field})', base)


def test_page_checks_apply_to_pages_without_entries():
    checks = CHECKS['page_content']({'purpose': 'Explain the project', 'entries': []})
    assert len(checks) == 3 and any(c['kind'] == 'judge' for c in checks)


def test_sft_index_keeps_program_groups_and_loads_unicode_rows(tmp_path):
    path = tmp_path / 'sft.jsonl'
    rows = [{'id': f'family-{i}-0', 'prompt': 'Grüße ' * 20, 'completion': str(i)} for i in range(3)]
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    indexed = index_pairs(path)
    held, train, _ = split_programs(indexed, 1)
    with path.open('rb') as f:
        for row in held + train:
            f.seek(row['offset'])
            assert json.loads(f.readline())['id'] == row['id']
    assert all('prompt' not in r for r in indexed)
