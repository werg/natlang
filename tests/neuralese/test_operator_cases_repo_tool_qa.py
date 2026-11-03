"""compact-value repo tool-result converter (scripts/neuralese_data/operator_cases_repo_tool_qa.py) on inline rows."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import operator_cases_repo_tool_qa as rq  # noqa: E402
from neuralese_data.common import Reject  # noqa: E402
from neuralese_data.records import validate  # noqa: E402

BODY = '\n'.join(f'def helper_{i}(value):\n    return value + {i}\n' for i in range(60))
Q = 'What is the signature of `load_model` in `pkg/rivagan.py`? Answer with the exact `def` line.'


def row(answer='def load_model(cls):', question=Q, slot=None, prompt='read pkg/rivagan.py', task='repo_qa:abc'):
    slot = slot if slot is not None else f'import os\n\nclass Riva:\n    {answer}\n        pass\n\n{BODY}'
    instruction = [{'role': 'system', 'content': 'sys'},
                   {'role': 'user', 'content': 'Repository overview (the whole file tree, compressed):\n\n' + question}]
    return {'family': 'agent_tool_slots', 'framing': 'tool_slots', 'context': json.dumps(['{"repo_overview": "x"}', slot]),
            'prompt': json.dumps(['', f'{prompt}: {question}']), 'instruction': json.dumps(instruction),
            'target': json.dumps([{'role': 'assistant', 'content': f'Found it.\nANSWER: {answer}'}]), 'split': 'train',
            'meta': json.dumps({'task_id': task, 'repo': 'o/r'})}


def test_convert_row_strips_overview_prefix_and_seeds_answer_lines():
    record = rq.convert_row(5, row(), {})
    assert validate(record) == []
    assert record['consumer']['context'][0]['content'] == Q
    assert record['target']['value'] == 'def load_model(cls):' and record['sources'][0]['meta']['tool'] == 'read'
    assert 'def load_model(cls):' in record['sources'][0]['meta']['operator_seed']['text']
    assert record['split_groups'] == ['bgkit-task:repo_qa:abc', 'repo:o/r']


@pytest.mark.parametrize('bad, reason', [
    (dict(slot='short'), 'answer-not-in-tool-results'),
    (dict(answer='None', slot='None ' * 400), 'trivial-answer'),
    (dict(task='other:1'), 'not-repo-qa'),
])
def test_rejects(bad, reason):
    with pytest.raises(Reject, match=reason):
        rq.convert_row(0, row(**bad), {})


def test_small_answer_slot_is_a_size_reject():
    with pytest.raises(Reject, match='source-size'):
        rq.convert_row(0, row(slot='def load_model(cls): pass'), {})
