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


def _persist_split(directory, partitions=('train', 'train', 'development', 'confirmation')):
    groups = sorted({json.loads(line)['group'] for line in (directory / 'tune.examples.jsonl').read_text().splitlines()})
    (directory / '.natlang' / 'tune').mkdir(parents=True)
    (directory / '.natlang' / 'tune' / 'splits.json').write_text(json.dumps(
        {'schema': 'natlang.tune-split/v1', 'components': {group: partitions[i % len(partitions)] for i, group in enumerate(groups)}}))
    return groups


def test_privileged_references_cover_training_roots_only_with_exact_stage_values(tmp_path):
    import pytest
    supply.write_programs(tmp_path, domains=['access-request', 'refund-decision'], variants=1, worlds=8)
    with pytest.raises(ValueError, match='persist the grouped split'):
        supply.write_privileged(tmp_path)
    for name in ('access-request', 'refund-decision'):
        _persist_split(tmp_path / name)
    manifest = supply.write_privileged(tmp_path)
    assert manifest['teacherCalls'] == 0 and len(manifest['programs']) == 2
    domains = {domain.name: domain for domain in supply.CATALOG}
    for item in manifest['programs']:
        directory = tmp_path / item['program']
        split = json.loads((directory / '.natlang' / 'tune' / 'splits.json').read_text())['components']
        rows = [json.loads(line) for line in (directory / supply.PRIVILEGED_FILE).read_text().splitlines()]
        assert item['partitions'] == {'train': 8, 'development': 4, 'confirmation': 4} and item['privileged_roots'] == 8
        assert {row['group'] for row in rows} == {group for group, partition in split.items() if partition == 'train'}
        program = domains[item['family']]
        examples = {json.dumps([row['group'], row['input']], sort_keys=True): row
                    for row in map(json.loads, (directory / 'tune.examples.jsonl').read_text().splitlines())}
        for row in rows:
            assert row['schema'] == 'natlang.tune-privileged/1'
            example = examples[json.dumps([row['group'], row['input']], sort_keys=True)]
            assert row['result'] == example['expected'] == row['calls'][-1]['value']
            names = [call['function'] for call in row['calls']]
            assert names[-1] == program.decide[0] and len(names) == 1 + bool(program.helper) + bool(program.classify)
    # access-request: its classifier's label, then the decision
    row = json.loads((tmp_path / 'access-request' / supply.PRIVILEGED_FILE).read_text().splitlines()[0])
    assert [call['function'] for call in row['calls']] == ['riskLevel', 'decideAccess']
    # Rewriting identical references is a no-op; changed references are refused (they bind their split).
    assert supply.write_privileged(tmp_path)['programs'] == manifest['programs']
    path = tmp_path / 'refund-decision' / supply.PRIVILEGED_FILE
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='exists with other references'):
        supply.write_privileged(tmp_path, ['refund-decision'])


def test_privileged_references_refuse_an_example_whose_oracle_changed(tmp_path):
    import pytest
    supply.write_programs(tmp_path, domains=['access-request'], variants=1, worlds=4)
    directory = tmp_path / 'access-request'
    _persist_split(directory, ('train',))
    lines = (directory / 'tune.examples.jsonl').read_text().splitlines()
    first = json.loads(lines[0])
    domain = next(domain for domain in supply.CATALOG if domain.name == 'access-request')
    first['expected'] = next(label for label in domain.labels if label != first['expected'])
    (directory / 'tune.examples.jsonl').write_text('\n'.join([json.dumps(first, sort_keys=True), *lines[1:]]) + '\n')
    with pytest.raises(ValueError, match='recomputed oracle'):
        supply.write_privileged(tmp_path)


def test_privileged_command_line(tmp_path, capsys):
    supply.write_programs(tmp_path, domains=['access-request'], variants=1, worlds=4)
    _persist_split(tmp_path / 'access-request', ('train', 'development'))
    assert supply.main(['privileged', '--programs', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {'command': 'privileged', 'programs': 1, 'privileged_roots': 4}
