"""Program supply: catalog programs are complete packages whose every example carries its exact oracle value."""
import json

from natlang_neuralese.tune import supply


def test_supply_writes_grouped_programs_with_exact_oracles(tmp_path):
    manifest = supply.write_programs(tmp_path, domains=['fraud-review', 'shipping-carrier'], variants=2, worlds=6)
    names = [item['program'] for item in manifest['programs']]
    assert names == ['shipping-carrier', 'shipping-carrier-v1', 'fraud-review', 'fraud-review-v1']
    assert manifest['families'] == ['fraud-review', 'shipping-carrier']
    domains = {domain.name: domain for domain in supply.CATALOG}
    for item in manifest['programs']:
        directory = tmp_path / item['program']
        assert (directory / 'main.nl').exists() and (directory / 'types.ts').exists()
        rows = [json.loads(line) for line in (directory / 'tune.examples.jsonl').read_text().splitlines()]
        assert len(rows) == 12 and len({row['group'] for row in rows}) == 6  # two invocations per world
        program = supply.variant(domains[item['family']], item['variant'])
        for row in rows:
            record = {key: value for key, value in row['input'][program.arg].items() if key != program.note}
            assert row['expected'] == supply.oracle(program, record)
    # A variant is a distinct program of the same family.
    assert (tmp_path / 'fraud-review' / 'main' / 'decidePayment.nl').read_text() != \
        (tmp_path / 'fraud-review-v1' / 'main' / 'decidePayment.nl').read_text() or \
        (tmp_path / 'fraud-review' / 'main' / 'riskScore.ts').read_text() != (tmp_path / 'fraud-review-v1' / 'main' / 'riskScore.ts').read_text()


def test_rules_render_record_fields_and_arguments():
    rules = [(('and', ('eq', 'flag', True), ('ge', 'score', 3)), 'a'), (('eq', 'level', supply.literal('high')), 'b')]
    text = supply.rules_text(rules, 'c', 'x.', fields={'flag': None, 'score': None})
    assert text == 'Apply in order: x.flag=true and x.score>=3 -> a; level=high -> b; otherwise -> c.'
    assert supply.decide(rules, 'c', {'flag': True, 'score': 3, 'level': 'low'}) == 'a'
    assert supply.decide(rules, 'c', {'flag': False, 'score': 3, 'level': 'high'}) == 'b'
