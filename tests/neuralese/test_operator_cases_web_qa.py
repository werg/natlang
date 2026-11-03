"""compact-value web-research converter (scripts/neuralese_data/operator_cases_web_qa.py) on inline rows."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import operator_cases_web_qa as wq  # noqa: E402
from neuralese_data.common import Reject  # noqa: E402
from neuralese_data.records import validate  # noqa: E402

FILLER = ' Unrelated background sentence about the archive and its catalogue.' * 20


def row(question='Who directed Rattle and Hum?', answer='Phil Joanou', gold=('Phil Joanou',), final=True, split='train',
        content=None, extra_slots=()):
    slots = ['{"compaction": 1, "child": null}', f'[https://a.org/x] Rattle and Hum\nRattle and Hum was directed by {answer}.{FILLER}',
             *extra_slots]
    prompts = ['compaction: ' + question, 'web search rattle and hum director: ' + question] + ['page x: ' + question] * len(extra_slots)
    meta = {'dataset': 'd', 'final': final, 'question': question, 'gold': list(gold), 'answer_match': 'em', 'trajectory_id': 't/1'}
    text = content if content is not None else f'ANSWER: **{answer}**'
    return {'family': 'web', 'framing': 'tool_slots', 'context': json.dumps(slots), 'prompt': json.dumps(prompts),
            'target': json.dumps([{'role': 'assistant', 'content': text, 'reasoning_content': 'think'}]),
            'split': split, 'meta': json.dumps(meta)}


def test_convert_row_drops_compaction_slots_and_seeds_checked_spans():
    record = wq.convert_row('web_openseeker_v3', 3, row())
    assert validate(record) == []
    assert 'compaction' not in record['sources'][0]['text'] and record['sources'][0]['role'] == 'tool_output'
    assert record['target']['value'] == 'Phil Joanou' and record['outcome']['checked'] == 'dataset-answer-em'
    seed = record['sources'][0]['meta']['operator_seed']
    assert seed['provenance'] == 'checked' and 'Phil Joanou' in seed['text']
    assert 'Phil Joanou' not in record['writer']['instructions'] + record['consumer']['context'][0]['content']
    assert record['split_groups'][0].startswith('web-q:')


@pytest.mark.parametrize('bad, reason', [
    (dict(final=False), 'not-final'),
    (dict(gold=('Somebody Else',)), 'answer-not-em'),
    (dict(content='ANSWER: Phil Joanou\nand more'), 'multi-line-answer'),
    (dict(answer='Joanou, again', gold=('Joanou, again',), content='ANSWER: Nobody At All', ), 'answer-not-em'),
])
def test_rejects(bad, reason):
    with pytest.raises(Reject, match=reason):
        wq.convert_row('web_sds_v3', 0, row(**bad))


def test_report_final_uses_gold_answer():
    record = wq.convert_row('web_miroverse_nc_v2', 1, row(content='# A long report without an answer line.'))
    assert record['target']['value'] == 'Phil Joanou' and record['outcome']['checked'] == 'dataset-gold-answer-em-report'
    unflagged = row(content='# report')
    unflagged['meta'] = json.dumps({**json.loads(unflagged['meta']), 'answer_match': None})
    with pytest.raises(Reject, match='no-final-answer'):
        wq.convert_row('web_miroverse_nc_v2', 2, unflagged)
