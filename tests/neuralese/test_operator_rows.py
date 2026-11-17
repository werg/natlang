"""Operator cases lower to exp0_sft per-call rows with the pass-through's import format (scripts/neuralese_data/operator_rows.py)."""
import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/neuralese_data'), str(ROOT / 'training/neuralese'), str(ROOT)]
LFM3 = glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*'))

DOC = ' '.join(f'Sentence {i} says the bridge on route {i} closed for repairs in spring.' for i in range(400))
BOILER = ('We did not uncover any evidence that the US copyright on this publication was renewed and this etext was '
          'produced from the magazine issue as printed')


def case(**over):
    record = {'id': 'case-1', 'split': 'train', 'split_groups': ['triviaqa-q:abc'],
              'sources': [{'role': 'tool_output', 'text': DOC}],
              'writer': {'instructions': 'Read this bash result for: which route closed', 'context': [],
                         'result_type': 'Neuralese<ToolResult>'},
              'consumer': {'context': [{'role': 'user', 'content': 'Which route closed first?'}],
                           'withheld': ['sources']},
              'target': {'kind': 'text', 'value': 'route 0'}, 'outcome': {'label': 'gold'}, 'lineage': {}}
    record.update(over)
    return record


def held_file(tmp_path, tasks):
    path = tmp_path / 'held.jsonl'
    path.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
    return path


def test_held_closure_matches_groups_ids_and_text_but_not_boilerplate(tmp_path):
    from operator_rows import HeldClosure
    question = 'Which famous rapid on the Chattooga River was featured in the 1972 film Deliverance by John Boorman?'
    held = HeldClosure.from_cases([held_file(tmp_path, [
        {'id': 'held-a', 'source_groups': ['triviaqa-q:xyz'], 'curriculum': {'split_group': 'triviaqa-q:xyz'},
         'semantics': {'inputs': {'question': question, 'doc': BOILER + ' First held article body text.'}}},
        {'id': 'held-b', 'source_groups': ['quality-article:2'],
         'semantics': {'inputs': {'doc': BOILER + ' Second held article with other words entirely.'}}}])])
    assert held.hit(case()) is None
    assert held.hit(case(split_groups=['triviaqa-q:xyz'])) == 'group'
    assert held.hit(case(id='held-a')) == 'id'
    asked = case(consumer={'context': [{'role': 'user', 'content': question}], 'withheld': ['sources']})
    assert held.hit(asked) == 'text'
    # Boilerplate shared by two held strings is not evidence of a shared source.
    assert held.hit(case(sources=[{'role': 'document', 'text': BOILER + ' ' + DOC}])) is None


@pytest.mark.skipif(not LFM3, reason='LFM3 tokenizer not cached')
def test_rows_carry_the_in_context_capture_and_the_consumer_import():
    from operator_rows import RowBuilder, load_tokenizer
    tokenizer, roles, fingerprint = load_tokenizer(sorted(LFM3)[-1])
    builder = RowBuilder(tokenizer, roles, fingerprint)
    rows = builder.build(case(), 'compact-value', 'k1')
    (op,), consumer = rows['operators'], rows['consumer']
    start, length, row = op['segs'][0]
    assert row['kind'] == 'operator' and len(row['sample']) == length == len(op['ids']) - start
    (at, key), = consumer['imports']
    doc = consumer['sources'][key]
    span = list(doc['ids'][doc['start']:doc['stop']])
    # The document is the operator's own site and reply prefix, then produce_open and the reply's content.
    reply = list(row['sample'])
    assert list(doc['ids'][:start]) == list(op['ids'][:start])
    assert doc['ids'][doc['start'] - 1] == roles[0]
    content_start, content_stop = builder.span_of(reply)
    assert span == reply[content_start:content_stop] and span
    assert tokenizer.convert_tokens_to_ids('</think>') not in span
    # The consumer reads the same tokens between the consume markers at the import rows.
    assert list(consumer['ids'][at:at + len(span)]) == span
    assert consumer['ids'][at - 1] == roles[2] and consumer['ids'][at + len(span)] == roles[3]
    assert 'Which route closed first?' in tokenizer.decode(consumer['ids'])
    cut = tokenizer.decode(rows['text_cut']['ids'])
    assert '<<cut off:' in cut and 'route 0' in tokenizer.decode(rows['text_cut']['segs'][0][2]['sample'])
    # The site is view's write site (one summarizer family).
    from natlang_neuralese.view import INSTRUCTIONS
    assert INSTRUCTIONS.split('\n')[0] in tokenizer.decode(op['ids'][:start])


@pytest.mark.skipif(not LFM3, reason='LFM3 tokenizer not cached')
def test_rows_feed_the_pass_through_plan():
    import numpy as np
    from operator_rows import RowBuilder, load_tokenizer
    import lfm3_fast_train
    tokenizer, roles, fingerprint = load_tokenizer(sorted(LFM3)[-1])
    rows = RowBuilder(tokenizer, roles, fingerprint).build(case(), 'compact-value', 'k1')
    consumer = rows['consumer']
    plan = lfm3_fast_train.passthrough_plan([(0, consumer['imports'], consumer['sources'])], len(consumer['ids']))
    assert not plan['fixed'] and len(plan['documents']) == 1
    assert np.array_equal(np.asarray(plan['documents'][0]), consumer['sources']['k1:0']['ids'])


def _consumer(task, span, before=(1, 2), after=(3, 4), target=(5, 6)):
    import numpy as np
    key = f'{task}:0'
    prompt = [*before, 90, *span, 91, *after]
    document = {'ids': np.asarray([7, 8, 80, *span], np.int32), 'start': 3, 'stop': 3 + len(span)}
    return {'ids': np.asarray([*prompt, *target], np.int32),
            'segs': [(len(prompt), len(target), {'kind': 'consumer', 'sample': np.asarray(target, np.int32)})],
            'imports': [(len(before) + 1, key)], 'sources': {key: document}, 'task': task, 'round': 0}


def test_ablation_arms_splice_donor_spans_and_standalone_documents():
    import numpy as np
    from operator_ablation import paired, shuffled, standalone
    a, b = _consumer('a', [11, 12]), _consumer('b', [21, 22, 23])
    sa, sb = shuffled([a, b])
    # a reads b's span (another length): the import rows, the target segment and the source move with it.
    assert list(sa['ids']) == [1, 2, 90, 21, 22, 23, 91, 3, 4, 5, 6]
    assert sa['segs'][0][0] == 9 and sa['imports'] == [(3, 'a:0')]
    assert list(sa['sources']['a:0']['ids']) == [7, 8, 80, 21, 22, 23]
    assert list(sb['ids'][3:5]) == [11, 12]
    alone = standalone(a, produce_open=80)
    assert list(alone['sources']['a:0']['ids']) == [80, 11, 12] and alone['sources']['a:0']['start'] == 1
    assert list(alone['ids']) == list(a['ids'])
    result = paired([1., 1.2, 0.8, 1.1])
    assert result['mean'] > 0 and result['better_share'] == 1.0 and result['ci95'][0] > 0
