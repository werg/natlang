import hashlib
import json
from scripts.neuralese_data.audit_acquired_source_occurrence_overlap import audit


def write(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_source_occurrence_detects_new_renderer_ids_and_internal_repetition(tmp_path):
    ids=write(tmp_path/'ids.json',['text:a:source-occurrence:new','text:b:source-occurrence:x','text:b:source-occurrence:y'])
    family=write(tmp_path/'families.json',{'source_sha256':'source','rows':[{'id':'text:a:source-occurrence:old'}]})
    predecessor=write(tmp_path/'predecessor.json',{'source':{'sha256':'source'},'rows':1,'family_map':family})
    screen=write(tmp_path/'screen.json',{'schema':'natlang.new-acquired-source-screen/1','training_admission':False,
        'prepared_source_sha256':'source','active_ids':ids,'active':{'rows':3}})
    result=audit(screen,predecessor,[],tmp_path/'overlap.json')
    assert result['predecessor_source_occurrence_overlaps']==['text:a']
    assert result['duplicate_source_occurrences_within_active']==1
