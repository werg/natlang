"""The Neuralese-declaring variant of a pool task keeps its inputs, oracle, reference and custody groups."""
import json

from scripts.rl_tasks.common import make_case
from scripts.rl_tasks.neuralese_variants import AUTHORED, helper_file, variant

NEURALESE = {'helper': 'read_part', 'args': {'part': 'string', 'question': 'string'}, 'returns': 'Neuralese<string>',
             'instruction': 'Write a private note of what part says about question.',
             'root_note': 'read_part gives a private note of one part. You may read the parts and answer from the notes.'}


def case(framing):
    return make_case(source='demo', family='reading', shape='demo-shape', upstream='demo/ds', upstream_id='item-1',
                     upstream_split='train', unit='unit-1', groups=['demo:unit-1'], root_name='answer',
                     args={'passage': 'string', 'question': 'string'}, returns='string',
                     instruction='Answer question from passage.', inputs={'passage': 'p', 'question': 'q'},
                     expected='x', oracle={'kind': 'exact'}, level=1, license_='MIT', wrong=['y'], neuralese=NEURALESE,
                     framing=framing)


def test_variant_is_the_generators_neuralese_framing():
    plain, framed = case('plain'), case('neuralese-helper')
    out = variant(plain, 'read_part.nl', framed['semantics']['files']['answer/read_part.nl'], NEURALESE['root_note'])
    assert out['id'] == framed['id'] != plain['id']
    assert out['semantics']['files'] == framed['semantics']['files']
    for key in ('inputs', 'expected', 'oracle'):
        assert out['semantics'][key] == plain['semantics'][key]
    assert out['curriculum']['reference'] == plain['curriculum']['reference']
    assert out['curriculum']['split_group'] == plain['curriculum']['split_group']
    assert out['source_groups'] == plain['source_groups'] and out['split'] == plain['split']
    ext = out['curriculum']['external']
    assert ext['framing'] == 'neuralese-helper' and ext['variant_of'] == plain['id']
    assert ext['instruction_sha256'] == framed['curriculum']['external']['instruction_sha256']
    assert plain['curriculum']['external']['framing'] == 'plain'   # the original is not modified


def test_authored_helpers_declare_neuralese():
    for name, args, text, note in AUTHORED.values():
        assert 'returns: "Neuralese<string>"' in helper_file(args, text)
        assert note.startswith(name) and json.dumps(args)
