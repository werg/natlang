import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import assemble_neuralese_recurrence as assembler


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def row(ident, split, group, *, family='authored_semantic_reducers', version=7, approved=True):
    groups=[group]
    return {'id':ident,'split':split,'source_groups':groups,'messages':[],'target':[],
        'task':{'program_ir':{'source_groups':groups,'license':'project-generated',
            'gold_sources':['constructed-world-oracle'],
            'curriculum':{'family':family}}},
        'neuralese_conversion':{'version':f'natlang.neuralese-conversion/{version}'},
        'training_admission':{'approved':approved},'outcome':{'accepted':True,
            'oracle':{'level':'exact','accepted':True}},'trace_admission':{'admitted':True}}


def review_for(records, pieces, *, allow_ids=(), allow_groups=(), hold_ids=(), hold_groups=()):
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def key(path):
        try:return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:return str(path.resolve())
    records=[records] if isinstance(records,Path) else list(records)
    pieces=[pieces] if isinstance(pieces,Path) else list(pieces)
    return {'schema':assembler.SOURCE_REVIEW_SCHEMA,
        'inputs':{'records':[{'path':key(path),'sha256':sha(path)} for path in records],
                  'pieces':[{'path':key(path),'sha256':sha(path)} for path in pieces]},
        'allow':{'source_groups':list(allow_groups),'target_ids':list(allow_ids)},
        'hold':{'source_groups':list(hold_groups),'target_ids':list(hold_ids)}}


def reviewed_native_row(ident, split, group, *, family='authored_semantic_source_worlds_v11'):
    result=row(ident,split,group,family=family,version=8)
    source_sha='a'*64; target_sha='b'*64
    result['source_ref']={'trajectory_id':ident,'source_row_sha256':source_sha,'native_target_sha256':target_sha}
    result['teacher_trajectory_id']=ident
    result['decision']={'index':0,'training_approved':True}
    result['training_admission']={'approved':True,'kind':'reviewed-native-decision','semantic_review':{
        'schema':'natlang.native-decision-approval/1','trajectory_id':ident,'source_row_sha256':source_sha,
        'decision_index':0,'target_sha256':target_sha,'review_sha256':'c'*64,
        'reason':'exact native action independently reviewed','evidence':['source artifact confirms the action value']}}
    result['outcome']={'accepted':False,'oracle':{'level':'exact','accepted':False}}
    return result


def stub_audit(monkeypatch):
    def run(command,check):
        out=Path(command[command.index('--out')+1])
        out.write_text(json.dumps({'structurally_closed':True,'records':1}))
    monkeypatch.setattr(assembler.subprocess,'run',run)


def test_v7_nonlegacy_family_requires_exact_source_review_and_does_not_admit_siblings(tmp_path,monkeypatch):
    records=tmp_path/'neuralese.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    rows=[row('allow-train','train','r4:task'),row('allow-test','test','v25:task'),
          row('same-family-sibling','train','unreviewed:task')]
    write_jsonl(records,rows)
    review=review_for(records,pieces,allow_ids=['allow-train','allow-test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    assert admitted==['allow-train','allow-test']
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert {item['id']:item['reason'] for item in rejected}=={
        'same-family-sibling':'source family or target not explicitly reviewed'}
    report=json.loads((out/'admission.json').read_text())
    assert report['explicitly_reviewed_targets']==['allow-test','allow-train']
    assert report['source_review']['sha256']==hashlib.sha256(review_path.read_bytes()).hexdigest()


def test_v10r2_source_world_family_requires_exact_target_ids_not_group_review(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    rows=[row('v31-allow-train','train','v31:train',family='authored_semantic_source_worlds_v10r2'),
          row('v31-group-sibling','test','v31:group',family='authored_semantic_source_worlds_v10r2'),
          row('baseline-test','test','baseline:test',family='decision_skill_catalog')]
    write_jsonl(records,rows)
    review=review_for(records,pieces,allow_ids=['v31-allow-train','baseline-test'],allow_groups=['v31:group'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    assert admitted==['v31-allow-train','baseline-test']
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert rejected==[{'id':'v31-group-sibling','reason':'source family or target not explicitly reviewed'}]


def test_versioned_source_world_families_accept_only_exact_ids_with_reviewed_native_proof(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    rows=[reviewed_native_row('v11-reviewed','train','v11:one'),
          reviewed_native_row('v12r3-reviewed','test','v12:one',family='authored_semantic_source_worlds_v12r3'),
          row('v12-group-sibling','train','v12:group',family='authored_semantic_source_worlds_v12r3'),
          reviewed_native_row('lookalike-unreviewed','train','lookalike:source',family='authored_semantic_source_worlds_v12x')]
    write_jsonl(records,rows)
    review=review_for(records,pieces,allow_ids=['v11-reviewed','v12r3-reviewed','lookalike-unreviewed'],
                      allow_groups=['v12:group'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    assert admitted==['v11-reviewed','v12r3-reviewed']
    rejected={r['id']:r['reason'] for r in map(json.loads,(out/'held-targets.jsonl').read_text().splitlines())}
    assert rejected=={'v12-group-sibling':'source family or target not explicitly reviewed',
                      'lookalike-unreviewed':'source family or target not explicitly reviewed'}


@pytest.mark.parametrize('tamper',['source-hash','target-proof','review-digest','empty-evidence'])
def test_reviewed_native_approval_rejects_stale_or_malformed_proof(tmp_path,monkeypatch,tamper):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    bad=reviewed_native_row('reviewed-train','train','reviewed:train')
    train=row('ordinary-train','train','ordinary:train',family='decision_skill_catalog')
    test=row('ordinary-test','test','ordinary:test',family='decision_skill_catalog')
    if tamper=='source-hash':bad['training_admission']['semantic_review']['source_row_sha256']='d'*64
    elif tamper=='target-proof':bad['source_ref']['native_target_sha256']='d'*64
    elif tamper=='review-digest':bad['training_admission']['semantic_review']['review_sha256']='stale'
    else:bad['training_admission']['semantic_review']['evidence']=[]
    write_jsonl(records,[bad,train,test])
    review=review_for(records,pieces,allow_ids=['reviewed-train','ordinary-train','ordinary-test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    assert admitted==['ordinary-train','ordinary-test']
    held=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert held==[{'id':'reviewed-train','reason':'reviewed native decision approval invalid'}]


def test_reviewed_native_conflicting_source_review_is_rejected(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    write_jsonl(records,[reviewed_native_row('conflict-train','train','conflict:train'),
                         row('ordinary-test','test','ordinary:test',family='decision_skill_catalog')])
    review=review_for(records,pieces,allow_ids=['conflict-train','ordinary-test'],hold_ids=['conflict-train'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='cannot both allow and hold'):
        assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),
                        '--out',str(tmp_path/'assembled')])


def test_failed_action_cannot_use_reviewed_native_approval_to_pass(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    failed=reviewed_native_row('failed-action-train','train','failed:train')
    failed['training_admission']['approved']=False
    failed['decision']['training_approved']=False
    failed['decision']['validation']={'reasons':['parent-action-failed']}
    failed['outcome']['action_ledger']=[{'status':'error'}]
    train=row('ordinary-train','train','ordinary:train',family='decision_skill_catalog')
    test=row('ordinary-test','test','ordinary:test',family='decision_skill_catalog')
    write_jsonl(records,[failed,train,test])
    review=review_for(records,pieces,allow_ids=['failed-action-train','ordinary-train','ordinary-test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    assert [json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]==['ordinary-train','ordinary-test']
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert rejected==[{'id':'failed-action-train','reason':'reviewed native decision approval invalid'}]


def test_source_world_without_reviewed_decision_proof_still_needs_accepted_parent_oracle(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    source_world=row('automatic-train','train','auto:source',family='authored_semantic_source_worlds_v12')
    source_world['outcome']={'accepted':False,'oracle':{'level':'exact','accepted':False}}
    source_world['training_admission']['kind']='exact-native-runtime-oracle'
    source_world['decision']={'index':0,'training_approved':True}
    train=row('ordinary-train','train','ordinary:train',family='decision_skill_catalog')
    test=row('ordinary-test','test','ordinary:test',family='decision_skill_catalog')
    write_jsonl(records,[source_world,train,test])
    review=review_for(records,pieces,allow_ids=['automatic-train','ordinary-train','ordinary-test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    assert [json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]==['ordinary-train','ordinary-test']
    held=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert held==[{'id':'automatic-train','reason':'runtime outcome not accepted'}]


def test_conversion7_remains_enabled_for_legacy_reviewed_families_without_new_manifest(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    write_jsonl(records,[row('legacy-train','train','authored-bounded-decisions-v1:a',family='decision_skill_catalog'),
                         row('legacy-test','test','authored-bounded-decisions-v1:b',family='decision_extract_chain')])
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--out',str(out)])
    assert len((out/'records.jsonl').read_text().splitlines())==2


@pytest.mark.parametrize('tamper', ['digest','stale-target'])
def test_source_review_rejects_stale_artifact_or_selectors(tmp_path,monkeypatch,tamper):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    write_jsonl(records,[row('allow-train','train','r4:task'),row('allow-test','test','v25:task')])
    review=review_for(records,pieces,allow_ids=['allow-train','allow-test'])
    if tamper=='stale-target':review['allow']['target_ids'][0]='removed-target'
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    if tamper=='digest':records.write_text(records.read_text()+'\n')
    out=tmp_path/'assembled'
    with pytest.raises(ValueError,match='stale|SHA-256'):
        assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])


@pytest.mark.parametrize(('gate','expected_reason'),[
    ('training','target not positively admitted'),
    ('runtime','runtime outcome not accepted'),
    ('exact-oracle','requires accepted exact oracle'),
    ('trace','trace not admitted'),
    ('source','unreviewed license or oracle')])
def test_source_review_cannot_bypass_positive_admission_gates(tmp_path,monkeypatch,gate,expected_reason):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    bad=row('bad','train','r4:task')
    if gate=='training':bad['training_admission']['approved']=False
    elif gate=='runtime':bad['outcome']['accepted']=False
    elif gate=='exact-oracle':bad['outcome']['oracle']['accepted']=False
    elif gate=='trace':bad['trace_admission']['admitted']=False
    elif gate=='source':bad['task']['program_ir']['license']='unknown'
    good_train=row('good-train','train','r4-good:task')
    good=row('good','test','v25:task')
    write_jsonl(records,[bad,good_train,good])
    review=review_for(records,pieces,allow_ids=['bad','good-train','good'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert admitted==['good-train','good']
    assert rejected[0]['reason']==expected_reason


def test_source_review_hold_wins_over_an_allowed_source_group(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    allowed=row('allowed','train','shared:source')
    held=row('held','train','shared:source')
    other=row('test','test','test:source')
    write_jsonl(records,[allowed,held,other])
    review=review_for(records,pieces,allow_ids=['test'],allow_groups=['shared:source'],hold_ids=['held'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert admitted==['allowed','test']
    assert rejected==[{'id':'held','reason':'explicit semantic source hold'}]


def test_source_group_split_leakage_still_rejects_reviewed_targets(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    write_jsonl(records,[row('train','train','shared:source'),row('test','test','shared:source')])
    review=review_for(records,pieces,allow_groups=['shared:source'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='source group crosses splits'):
        assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),
                        '--out',str(tmp_path/'assembled')])


def test_reviewed_reader_still_requires_producer_fixed_point_closure(tmp_path,monkeypatch):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl';pieces.write_text('')
    reader=row('reader','train','reader:source')
    reader['messages']=[{'role':'user','content':[{'type':'read','name':'missing-write'}]}]
    rows=[reader,row('independent-train','train','independent:train'),
          row('test','test','independent:test'),row('unreviewed-producer','train','producer:source')]
    rows[-1]['target']={'value':{'$write':{'name':'missing-write'}}}
    write_jsonl(records,rows)
    review=review_for(records,pieces,allow_ids=['reader','independent-train','test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',str(records),'--pieces',str(pieces),'--source-review',str(review_path),'--out',str(out)])
    admitted=[json.loads(line)['id'] for line in (out/'records.jsonl').read_text().splitlines()]
    rejected=[json.loads(line) for line in (out/'held-targets.jsonl').read_text().splitlines()]
    assert admitted==['independent-train','test']
    assert any(item['id']=='reader' and item['reason']=='producer closure' for item in rejected)


def test_conflicting_soft_prompt_namespaces_by_content_after_original_hash_review(tmp_path,monkeypatch):
    record_paths=[tmp_path/'old-records.jsonl',tmp_path/'new-records.jsonl']
    piece_paths=[tmp_path/'old-pieces.jsonl',tmp_path/'new-pieces.jsonl']
    old=row('old-train','train','old:source')
    new=row('new-test','test','new:source')
    old['messages']=[{'role':'system','content':[{'type':'soft','name':'prompt:interpreter'},
        {'type':'read','name':'graph-value'}]}]
    new['messages']=[{'role':'system','content':[{'type':'soft','name':'prompt:interpreter'},
        {'type':'read','name':'graph-value'}]}]
    old['target']={'$write':{'name':'graph-value'}}
    new['target']={'$write':{'name':'graph-value'}}
    write_jsonl(record_paths[0],[old]);write_jsonl(record_paths[1],[new])
    old_text='Older tool instructions; exact bytes retained.'
    new_text='Newer tool instructions; exact bytes retained.'
    write_jsonl(piece_paths[0],[{'name':'prompt:interpreter','kind':'system-prompt','text':old_text}])
    write_jsonl(piece_paths[1],[{'name':'prompt:interpreter','kind':'system-prompt','text':new_text}])
    review=review_for(record_paths,piece_paths,allow_ids=['old-train','new-test'])
    review_path=tmp_path/'review.json';review_path.write_text(json.dumps(review))
    out=tmp_path/'assembled';stub_audit(monkeypatch)
    assembler.main(['--records',*[str(p) for p in record_paths],'--pieces',*[str(p) for p in piece_paths],
        '--source-review',str(review_path),'--out',str(out)])
    rows=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    pieces={p['name']:p['text'] for p in map(json.loads,(out/'pieces.jsonl').read_text().splitlines())}
    names_by_id={r['id']:r['messages'][0]['content'][0]['name'] for r in rows}
    assert names_by_id['old-train']!=names_by_id['new-test']
    assert pieces[names_by_id['old-train']]==old_text
    assert pieces[names_by_id['new-test']]==new_text
    assert [r['target'] for r in rows]==[old['target'],new['target']]
    transform=json.loads((out/'soft-piece-namespace-transform.json').read_text())
    assert transform['applied'] is True
    assert len(transform['collisions'])==1
    assert len(transform['inputs'])==2
    report=json.loads((out/'admission.json').read_text())
    assert report['source_review']['sha256']==hashlib.sha256(review_path.read_bytes()).hexdigest()
    assert report['inputs'][str(record_paths[0])]==hashlib.sha256(record_paths[0].read_bytes()).hexdigest()


def test_colliding_soft_name_cannot_also_be_graph_name(tmp_path):
    record_paths=[tmp_path/'a.jsonl',tmp_path/'b.jsonl']
    piece_paths=[tmp_path/'a-pieces.jsonl',tmp_path/'b-pieces.jsonl']
    a=row('a','train','a:source');b=row('b','test','b:source')
    a['messages']=[{'role':'system','content':[{'type':'soft','name':'shared'}]},
                   {'role':'user','content':[{'type':'read','name':'shared'}]}]
    b['messages']=[{'role':'system','content':[{'type':'soft','name':'shared'}]}]
    write_jsonl(record_paths[0],[a]);write_jsonl(record_paths[1],[b])
    write_jsonl(piece_paths[0],[{'name':'shared','kind':'text','text':'one'}])
    write_jsonl(piece_paths[1],[{'name':'shared','kind':'text','text':'two'}])
    with pytest.raises(ValueError,match='also used as graph read/write'):
        assembler.load_assembly_inputs(record_paths,piece_paths)


def test_conflicting_piece_inputs_require_one_to_one_pairing(tmp_path):
    records=tmp_path/'records.jsonl';pieces=[tmp_path/'one.jsonl',tmp_path/'two.jsonl']
    write_jsonl(records,[row('a','train','a:source')])
    for p in pieces:write_jsonl(p,[{'name':'p','kind':'text','text':str(p)}])
    with pytest.raises(ValueError,match='paired one-to-one'):
        assembler.load_assembly_inputs([records],pieces)


def test_same_input_cannot_define_two_different_payloads_for_piece_name(tmp_path):
    records=tmp_path/'records.jsonl';pieces=tmp_path/'pieces.jsonl'
    write_jsonl(records,[row('a','train','a:source')])
    write_jsonl(pieces,[{'name':'p','kind':'text','text':'one'},
                        {'name':'p','kind':'text','text':'two'}])
    with pytest.raises(ValueError,match='one input contains multiple differing pieces'):
        assembler.load_assembly_inputs([records],[pieces])
