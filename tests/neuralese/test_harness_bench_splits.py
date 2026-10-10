"""Harness-bench splits follow the published corpora's cross-corpus closure (records.placements; VIEW_CORPUS.md §4.1)."""
import json

from natlang_neuralese.harness_bench import records
from natlang_neuralese.harness_bench.openhands import normalize

from test_harness_bench_openhands import ROW


def published_index(root, corpus_id, groups):
    root.mkdir(parents=True)
    (root / 'index.json').write_text(json.dumps({'corpus_id': corpus_id, 'records': len(groups)}))
    (root / 'groups.json').write_text(json.dumps(groups))
    return root


def test_published_repositories_keep_their_split_and_others_use_split_of(tmp_path):
    own = {repo: records.split_of(repo, 5) for repo in ('acme/lib', 'other/one', 'free/repo', 'mixed/up')}
    flip = {'train': 'test', 'test': 'train'}
    s1 = published_index(tmp_path / 's1', 's1', {
        'repo:acme/lib': flip[own['acme/lib']],                 # S1's repo key, case-folded
        'swe-instance:other__one-3': flip[own['other/one']],   # reached through an instance only
        'repo:mixed/up': 'train', 'swe-instance:mixed__up-2': 'test'})
    pairs = [('Acme/Lib', 'Acme__Lib-7'), ('other/one', 'other__one-3'), ('other/one', 'other__one-4'),
             ('free/repo', 'free__repo-1'), ('mixed/up', 'mixed__up-2')]
    placed, report = records.placements(pairs, [s1], 5)
    assert placed['Acme/Lib'] == {'split': flip[own['acme/lib']], 'rule': 'published', 'touched': {'s1': [flip[own['acme/lib']]]}}
    assert placed['other/one']['split'] == flip[own['other/one']]  # the whole repository, both instances
    assert placed['free/repo'] == {'split': own['free/repo'], 'rule': 'split_of'}
    assert placed['mixed/up']['split'] is None and report['unplaceable'] == ['mixed/up']
    assert report['published'] == 2 and report['split_of'] == 1 and report['indexes'][0]['corpus_id'] == 's1'


def test_build_records_the_placement():
    transcript = normalize(ROW, 'pi')
    kwargs = dict(system_piece='prompt:pi-agent#sha256:x', surface_sha='s', tools=[], corpus='c', test_percent=5,
                  view_chars=2000, preview_chars=40000, targets='all', per_trajectory=8, seed=0, transcript=transcript)
    default = list(records.build(None, **kwargs))
    assert default and all(r['split'] == records.split_of('acme/lib', 5) for r in default)
    assert default[0]['provenance']['split_placement']['rule'] == 'split_of'
    placement = {'split': 'test', 'rule': 'published', 'touched': {'s1': ['test']}}
    placed = list(records.build(None, placement=placement, **kwargs))
    assert all(r['split'] == 'test' and r['provenance']['split_placement'] == placement for r in placed)
    try:
        list(records.build(None, placement={'split': None, 'rule': 'unplaceable'}, **kwargs))
    except ValueError as error:
        assert 'several published splits' in str(error)
    else:
        raise AssertionError('an unplaceable repository must not be built')
