"""Operator cases lower to exp0_sft per-call rows with the pass-through's import format (scripts/neuralese_data/operator_rows.py)."""
import glob
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/neuralese_data'), str(ROOT / 'training/neuralese'), str(ROOT)]
LFM3 = glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*'))
NODE = shutil.which('node') or str(Path.home() / '.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node')
RUNTIME = pytest.mark.skipif(not LFM3 or not (ROOT / 'ts-host/dist/native/operator-site.js').exists()
                             or not Path(NODE).exists(), reason='LFM3 tokenizer, node or the built ts-host dist missing')

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


@RUNTIME
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
    # Seeds and exact answers carry no reasoning: their empty think block is context, the reply after it the target.
    close = tokenizer.convert_tokens_to_ids('</think>')
    for chain in (op, consumer, rows['text_cut']):
        (_, _, r), = chain['segs']
        assert r['sample'][0] == close and r['weight'][0] == 0 and r['weight'][-1] > 0
    kept = RowBuilder(tokenizer, roles, fingerprint, think='train').build(case(), 'compact-value', 'k1')
    assert 'weight' not in kept['consumer']['segs'][0][2] and kept['operators'][0]['segs'][0][2]['weight'][0] == 1
    # The consumer reads the same tokens between the consume markers at the import rows.
    assert list(consumer['ids'][at:at + len(span)]) == span
    assert consumer['ids'][at - 1] == roles[2] and consumer['ids'][at + len(span)] == roles[3]
    shown = tokenizer.decode(consumer['ids'])
    assert 'Which route closed first?' in shown and 'Compacted value: ' in shown and 'transcript.entry(0).output' in shown
    cut = tokenizer.decode(rows['text_cut']['ids'])
    assert '<<cut off:' in cut and 'route 0' in tokenizer.decode(rows['text_cut']['segs'][0][2]['sample'])
    # The site is the runtime's: view's write site with the runtime's compact-value instructions.
    from natlang_neuralese.view import INSTRUCTIONS
    site = tokenizer.decode(op['ids'][:start])
    assert INSTRUCTIONS.split('\n')[0] in site and 'Read this bash result for: which route closed' in site


@RUNTIME
def test_runtime_sites_render_arguments_as_the_listing_shows_them():
    from operator_rows import RuntimeSites, site_request
    sites = RuntimeSites()
    try:
        record = case(writer={'instructions': 'The call that receives this passage answers: which route', 'context': []})
        site = sites.site(site_request('compact-params', record, {'text': DOC}, 0, 1))
        assert site['compacted'] and site['operator'] == 'compact-params'
        assert site['messages'][1]['content'].startswith(
            'instructions:\nThe call that receives this argument as passage (string) answers: which route')
        assert site['messages'][1]['content'].endswith('value:\n' + json.dumps(DOC))
        assert site['state']['before'] == 'passage: string = ' and site['state']['after'].startswith('  // view of')
        assert site['text'].startswith('passage: string = ') and '<<cut off:' in site['text']
        small = sites.site(site_request('compact-params', record, {'text': 'short'}, 0, 1))
        assert not small['compacted'] and small['text'] == 'passage: string = "short"'
    finally:
        sites.close()


@RUNTIME
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


def test_merge_joins_train_files_and_caps_val_consumers(tmp_path):
    import json, pickle
    from types import SimpleNamespace
    from operator_rows import merge
    a, b = _consumer('a', [11, 12]), _consumer('b', [21])
    op = {**_consumer('c', [5]), 'segs': [(2, 2, {'kind': 'operator', 'sample': a['segs'][0][2]['sample']})]}
    for name, chains in (('train', [op, a]), ('test', [a, b])):
        pickle.dump({'chains': chains}, open(tmp_path / f'{name}.pkl', 'wb'))
    (tmp_path / 'manifest.json').write_text(json.dumps({'sources': {'compact-value/x': {'files': {
        'train': {'path': str(tmp_path / 'train.pkl')}, 'test': {'path': str(tmp_path / 'test.pkl')}}}}}))
    merge(SimpleNamespace(rows=str(tmp_path), out=str(tmp_path / 'm'), max_cases_per_source=0, val_per_source=1))
    summary = json.loads((tmp_path / 'm/merge.json').read_text())
    assert summary['chains']['chains'] == 2 and summary['val']['chains'] == 1
    assert pickle.load(open(tmp_path / 'm/val.pkl', 'rb'))['chains'][0]['task'] == 'a'
