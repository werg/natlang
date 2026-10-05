import json
from scripts.create_student_rewrite_pipeline import build


def test_chunk_search_uses_ordinary_render_and_audit_pipeline(tmp_path):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'schema':'natlang.student_chunk_rewrite_plan/1',
        'output':str(tmp_path/'collection'),'pins':{},'teacher_artifacts':[],
        'student':{'weight_pins':{}},'root_approved':True}))
    pipeline=build(plan,model='student',revision='a'*40,max_len=16384,python='python3')
    assert pipeline['method']=='verified-threshold-chunk-rewrite/1'
    command=pipeline['stages'][0]['command']
    assert command[1].endswith('/rewrite-student-chunks.mjs')
    assert command[3]=='--execute' and len(command[4])==64
    assert '--sha256' not in command
    assert pipeline['stages'][1]['command'][1].endswith('/render_training_corpus.py')
    assert pipeline['stages'][2]['command'][1].endswith('/audit_training_corpus.py')
    assert pipeline['automatic_training_publication'] is False
