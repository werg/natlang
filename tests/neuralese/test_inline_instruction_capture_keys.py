from training.neuralese.natlang_neuralese.train.inline_instructions import _capture_names_match


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
