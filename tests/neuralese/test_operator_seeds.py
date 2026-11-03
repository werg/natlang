"""The one shared deterministic seed helper of learned-operator producer spans (data/operator_seeds.py)."""
import json

import pytest

from natlang_neuralese.data.operator_seeds import (budget_chars, mechanical_seed, seed_for, supplied_seed)

DOC = ('The bridge was built in 1932 by the city engineer Anna Berg. It spans the Elbe river. '
       'Traffic was closed in 1990 for repairs. The museum opened later in Dresden.\n') * 40


def test_answer_lines_win_and_are_deterministic():
    one = mechanical_seed('extract', {'text': DOC}, question='When was the bridge built?', answers=('1932',))
    assert one == mechanical_seed('extract', {'text': DOC}, question='When was the bridge built?', answers=('1932',))
    assert one['provenance'] == 'mechanical:answer-lines' and '1932' in one['text']
    assert 'museum' not in one['text']


def test_overlap_then_identifiers_then_outline():
    overlap = mechanical_seed('compact-params', {'text': DOC}, question='who closed traffic for repairs')
    assert overlap['provenance'] == 'mechanical:overlap-sentences' and 'repairs' in overlap['text']
    code = 'def load(path):\n    return open("src/app/config.yaml").read()  # line 12\n' * 100
    value = mechanical_seed('compact-value', {'text': code})
    assert value['provenance'] == 'mechanical:identifiers-outline' and 'src/app/config.yaml' in value['text']
    data = json.dumps([{'id': i, 'name': f'n{i}'} for i in range(300)])
    outline = mechanical_seed('compact-history', {'text': data})
    assert outline['provenance'] == 'mechanical:outline' and 'JSON array of 300 items' in outline['text']
    assert len(outline['text']) <= budget_chars(len(data))


def test_history_outline_and_budget_scaling():
    messages = [{'role': 'assistant', 'content': 'Reading /repo/src/app.py now.',
                 'tool_calls': [{'function': {'name': 'read', 'arguments': '{"path": "/repo/src/app.py"}'}}]},
                {'role': 'tool', 'content': 'Traceback: ValueError in /repo/src/app.py:40'},
                {'role': 'assistant', 'content': 'Fix the parser next.'}]
    seed = mechanical_seed('compact-history', {'messages': messages})
    assert seed['provenance'] == 'mechanical:history-outline'
    assert '/repo/src/app.py' in seed['text'] and 'Fix the parser next.' in seed['text'] and 'read(' in seed['text']
    assert budget_chars(100) == 200 and budget_chars(10**7) == 2400 and budget_chars(10_000) == 800


def test_supplied_seeds_are_validated_and_preferred():
    source = {'text': DOC, 'meta': {'operator_seed': {'text': 'Anna Berg built it.', 'provenance': 'gold'}}}
    record = {'target': {'kind': 'text', 'value': 'Anna Berg'}, 'writer': {'instructions': 'who built it'},
              'consumer': {'context': []}}
    assert seed_for('extract', record, source)['text'] == 'Anna Berg built it.'
    assert seed_for('extract', record, {'text': DOC})['provenance'] == 'mechanical:answer-lines'
    with pytest.raises(ValueError):
        supplied_seed({'meta': {'operator_seed': {'text': 'x', 'provenance': 'summary-model'}}})
