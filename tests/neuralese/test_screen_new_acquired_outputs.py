import copy
import json
from types import SimpleNamespace

from scripts.neuralese_data import screen_new_acquired_outputs as screening
from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
from natlang_neuralese.data.native_sources import _row_bindings
from test_static_native_context import row,FP


def test_screen_reconciles_real_graphs_and_current_quality_groups(tmp_path,monkeypatch):
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return screening.bind(path)
    original=row();new=copy.deepcopy(original);new['id']='new';new['source_groups']=['new-group']
    excluded=copy.deepcopy(original);excluded['id']='excluded';excluded['source_groups']=['quality-group']
    pending=copy.deepcopy(original);pending['id']='pending';pending['source_groups']=['pending-group']
    graphs=[original,new,excluded,pending];path=tmp_path/'conversion'/'train'/'native-graphs.jsonl.zst';path.parent.mkdir(parents=True)
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    writer=GraphWriter(path)
    for graph in graphs:writer.write(graph)
    output=writer.finish()
    receipt={'schema':'natlang.acquired-full-session-graphs/1','training_admission':False,
             'expanded_tokenizer_fingerprint':FP,'outputs':[output]}
    write(path.parent.parent/'receipt.json',receipt)
    summary=write(tmp_path/'summary.json',{'schema':'natlang.acquired-session-student-conversion/1','incomplete':[],
        'by_source':{'D32':{'outputs':[output]}}})
    parent=write(tmp_path/'parent-coverage.json',{'receipts':[]});excluded_inventory=write(tmp_path/'excluded-coverage.json',{'receipts':[]})
    coverage=write(tmp_path/'coverage.json',{'receipts':[],'parent_inventory':parent,'excluded_inventory':excluded_inventory})
    binding=_row_bindings([indexed_graph_metadata(original,tokenizer_fingerprint=FP)])[original['id']]
    family=write(tmp_path/'family.json',{'source_sha256':'source-sha','rows':[{'id':original['id'],'binding':binding,'family':'ordinary'}]})
    partition=write(tmp_path/'partition.json',{'pending_rows':[{'id':'pending'}]})
    prior_screen=write(tmp_path/'prior-screen.json',{'pending_partition':partition})
    predecessor=write(tmp_path/'predecessor.json',{'rows':1,'family_map':family,'source':{'sha256':'source-sha'},'source_screen':prior_screen})
    protection_path=write(tmp_path/'quality.json',{})
    protection=SimpleNamespace(identities=set(),digest='digest')
    monkeypatch.setattr(screening.ProtectedWorlds,'load',lambda:protection)
    monkeypatch.setattr(screening,'current_protection',lambda path:{'held':set(),'quality':{'quality-group'},'pending':set(),'proof':{'sha256':'q'}})
    monkeypatch.setattr(screening,'load_decision_holds',lambda paths:(set(),[]))
    result=screening.screen(summary['path'],coverage['path'],predecessor['path'],protection_path['path'],tmp_path/'out')
    assert result['counts']=={'raw':4,'already-present-exact':1,'active':1,'current-source-quality-or-held-group':1,'already-pending-source-obligation':1}
    assert result['training_admission'] is False and result['active']['rows']==1
    from natlang_neuralese.common.jsonio import open_jsonl_text
    with open_jsonl_text(path) as source, open_jsonl_text(result['active']['path']) as screened:
        lines=list(source);assert list(screened)==[lines[1]]
    second=screening.screen(summary['path'],coverage['path'],predecessor['path'],protection_path['path'],tmp_path/'out2',
                            reserved_screens=[tmp_path/'out'/'receipt.json'])
    assert second['counts']['already-reserved-source-obligation']==1 and second['active']['rows']==0
