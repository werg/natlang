from training.neuralese.natlang_neuralese.train.inline_instructions import (
    _capture_names_match,
    _primitive_literal_matches,
)


def test_capture_prefix_accepts_attested_property_aliases_and_nested_values():
    assert _capture_names_match(
        'criterion, rule: input.decisionRule, bounds: limits[0], text: format(value, ",")',
        ['criterion', 'rule', 'bounds', 'text'],
    )
    assert _capture_names_match('"rule": input.rule', ['rule'])


def test_capture_prefix_rejects_non_binding_object_properties():
    assert not _capture_names_match('...captures, rule', ['rule'])
    assert not _capture_names_match('[key]: value', ['key'])
    assert not _capture_names_match('rule,', ['rule'])


def test_primitive_literal_capture_requires_exact_host_value_type_and_value():
    assert _primitive_literal_matches('3', 'number', 3)
    assert _primitive_literal_matches('"approved"', 'string', 'approved')
    assert _primitive_literal_matches('true', 'boolean', True)
    assert not _primitive_literal_matches('3', 'number', 4)
    assert not _primitive_literal_matches('3', 'string', '3')
    assert not _primitive_literal_matches('"approved" | "held"', 'string', 'approved')
    assert not _primitive_literal_matches('-0', 'number', 0.0)
