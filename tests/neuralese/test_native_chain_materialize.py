import hashlib

from natlang_neuralese.model.shared_state import ROLE_MARKERS
from scripts.neuralese_data.native_chain_materialize import materialize
from scripts.neuralese_data.native_chain_render import edge_digest


def test_joint_marker_intervals_preserve_distinct_equal_reads_and_export():
    class Tokenizer:
        def __call__(self, text, **_): return {'input_ids': [ord(c) for c in text]}
        def decode(self, ids, **_): return ''.join(chr(x) for x in ids)
    tok = Tokenizer(); body = 'same'
    edge = {'body_sha256': hashlib.sha256(body.encode()).hexdigest()}; eid = edge_digest(edge)
    writer_text = 'A' + ROLE_MARKERS[0] + body + ROLE_MARKERS[1] + 'Z'
    wrapped = ROLE_MARKERS[2] + body + ROLE_MARKERS[3]
    reader_text = 'Q' + wrapped + 'R' + wrapped + 'T'
    def row(text, suffix):
        return {'text': text, 'token_ids': [ord(c) for c in text],
                'token_offsets': [[i, i+1] for i in range(len(text))], 'supervised_suffix_start': suffix}
    writer = row(writer_text, 1); writer['producer_rendered_intervals'] = [
        {'char_start': 1+len(ROLE_MARKERS[0]), 'char_stop': 1+len(ROLE_MARKERS[0])+len(body)}]
    reader = row(reader_text, len(reader_text)-1)
    starts = [1, 2+len(wrapped)]
    reader['context_rendered_intervals'] = {str(i): {'char_start': start, 'char_stop': start+len(wrapped)}
                                             for i, start in enumerate(starts)}
    producer = {'body': body, 'writer_occurrence_id': 'w', 'edge_sha256': [eid]}
    reads = [{'edge_sha256': eid, 'replacement': {'render_tag': str(i), 'occurrence_id': f'r{i}',
        'pointer': f'/messages/{i}', 'replacement': {'text': wrapped}}} for i in range(2)]
    rendered = {'rows': {'w': writer, 'r': reader}, 'producer_entries': {'w': [producer], 'r': []},
                'reads': {'w': [], 'r': reads}}
    records = {'w': {'target': {'content': 'w'}}, 'r': {'target': {'content': 'r'}}}
    nodes = materialize(rendered, records, [edge], tokenizer=tok, role_ids=(300, 301, 302, 303))
    assert nodes['w']['producing'] == [[2, 6]]
    assert [(x['start'], x['stop']) for x in nodes['r']['reads']] == [(2, 6), (9, 13)]
    assert all(x['source_start'] == 2 and x['source_stop'] == 6 for x in nodes['r']['reads'])
    assert nodes['w']['supervised_suffix_start'] == 1
