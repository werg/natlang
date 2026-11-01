import copy
import json
from types import SimpleNamespace

import pytest

from scripts.neuralese_data import filter_acquired_predecessor_collisions as filtering
from scripts.neuralese_data.build_acquired_streaming_successor import bind, predecessor_collision_count
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text


def fixture(tmp_path, monkeypatch):
    digest='a'*64
    monkeypatch.setattr(filtering.ProtectedWorlds,'load',lambda:SimpleNamespace(digest=digest))
    def write(name, value):
        path=tmp_path/name;path.write_text(json.dumps(value));return path
    row={'schema':'natlang.neuralese-native-graph/1','id':'existing','split':'train',
         'source_groups':['world:existing'],'tokenizer_fingerprint':digest,
         'nodes':[{'name':'text','token_ids':[1,2],'target_weights':[0.,1.]}]}
    novel=copy.deepcopy(row);novel['id']='novel';novel['source_groups']=['world:novel']
    previous=copy.deepcopy(row);previous['nodes'][0]['token_ids']=[1,4]
    filtering._init(digest,(20,21,22,23))
    old_binding=filtering._parse_line(json.dumps(previous))[0]['binding']
    old_rows=[{'id':'existing','binding':old_binding}]
    family=write('family.json',{'source_sha256':'b'*64,'rows':old_rows})
    geometry=write('geometry.json',{'identity':{'tokenizer':digest,'roles':[20,21,22,23]}})
    predecessor=write('predecessor.json',{'rows':1,'protection_digest':digest,'source':{'sha256':'b'*64},
                        'family_map':bind(family),'geometry':bind(geometry)})
    writer=GraphWriter(tmp_path/'input.jsonl.zst')
    for item in (row,novel):writer.write(item)
    graph=writer.finish();graph['tokenizer_fingerprint']=digest
    screen=write('screen.json',{'schema':'natlang.acquired-current-protection-screen/1',
        'protected_worlds':{'identity_set_sha256':digest,'matches_expected':True},
        'coverage_audit':{},'input_receipts':{},'excluded_current_protection':[],
        'output':graph,'accounting':{'eligible_train_graphs':2}})
    signals=tmp_path/'signals.jsonl'
    signals.write_text(''.join(json.dumps({'id':r['id'],'graph_sha256':canonical_json_sha256_hex(r),
                                         'forward_tokens':2,'raw_positive_tokens':1})+'\n' for r in (row,novel)))
    census=write('census.json',{'screen':bind(screen),'row_signals':bind(signals),'rows':2})
    return predecessor,screen,census,old_rows


def test_changed_candidate_keeps_exact_parent_and_novel_rows_are_trainable(tmp_path,monkeypatch):
    predecessor,screen,census,old=fixture(tmp_path,monkeypatch)
    result=filtering.filter_source(predecessor,screen,census,tmp_path/'filtered')
    derived=json.loads((tmp_path/'filtered/receipt.json').read_text())
    proof=json.loads((tmp_path/'filtered/collision-filter.json').read_text())
    assert result['retained']==1 and result['collisions']==1
    assert proof['excluded'][0]['equivalent_training_binding'] is False
    assert predecessor_collision_count(derived,predecessor,old)==1
    with open_jsonl_text(tmp_path/'filtered/train/native-graphs.jsonl.zst') as f:
        rows=[json.loads(line) for line in f]
    assert [row['id'] for row in rows]==['novel']
    assert filtering._parse_line(json.dumps(rows[0]))[0]['supervised_tokens']==1
    assert proof['input_graph']==json.loads(screen.read_text())['output']
    with pytest.raises(ValueError,match='actual predecessor binding'):
        predecessor_collision_count(derived,predecessor,[{'id':'existing','binding':'c'*64}])


def test_census_tampering_aborts_compression_without_publishing(tmp_path,monkeypatch):
    predecessor,screen,census,old=fixture(tmp_path,monkeypatch)
    signals=tmp_path/'signals.jsonl';signals.write_text(signals.read_text().replace('existing','tampered'))
    value=json.loads(census.read_text());value['row_signals']=bind(signals);census.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='graph/census'):
        filtering.filter_source(predecessor,screen,census,tmp_path/'filtered')
    assert not (tmp_path/'filtered/receipt.json').exists()
    assert not list((tmp_path/'filtered/train').iterdir())
