import hashlib
import json

import pytest

from scripts.neuralese_data.review_acquired_converter_equivalence import review


def fixture(tmp_path):
    def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    builder="def session_graphs():\n common = {'source_record_ids': ['trajectory']}\n return common\n"
    files={}
    for key in ('builder','conversion'):
        old=tmp_path/(key+'-old.py');new=tmp_path/(key+'-new.py')
        if key=='builder':old.write_text(builder);new.write_text(builder+'\ndef archive_empty(): pass\n')
        else:
            for historical,path in ((True,old),(False,new)):
                source=''
                for whole,name in ((False,'text_row_graph'),(True,'whole_trajectory_text_row_graph')):
                    source+=f'def {name}(row):\n identifier = "id"\n source_records = [identifier]\n'
                    fallback='source_records' if whole else "row.get('source_record_ids') or [identifier]"
                    if historical:
                        source+=f" twin = row.get('source_twin_id') or {fallback}\n twin_id = canonical_json_sha256_hex(sorted(str(value) for value in twin))\n"
                    else:
                        source+=f" twin = _normalized_twin_identity(row.get('source_twin_id'), fallback={fallback})\n twin_id = canonical_json_sha256_hex(twin)\n"
                    source+=' return twin_id\n'
                path.write_text(source)
        files[key]={'historical':ref(old),'current':{**ref(new),'path':new.name}}
    registry=tmp_path/'review.json';registry.write_text(json.dumps({'schema':'natlang.acquired-converter-full-session-equivalence/1','files':files}))
    return registry


def test_reviewed_fallback_transition_is_accepted_without_executing_historical_code(tmp_path):
    registry=fixture(tmp_path);result=review(tmp_path,registry)
    assert result['registry']['path']==str(registry)
    assert set(result['files'])=={'builder','conversion'}


def test_even_rehashed_review_rejects_a_changed_actual_renderer(tmp_path):
    registry=fixture(tmp_path);value=json.loads(registry.read_text());path=tmp_path/value['files']['builder']['current']['path']
    path.write_text(path.read_text().replace('return common',"common['source_twin_id'] = 'unexpected'\n return common"))
    value['files']['builder']['current']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();registry.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='full-session renderer changed'):review(tmp_path,registry)
