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
    return {'schema':assembler.SOURCE_REVIEW_SCHEMA,
        'inputs':{'records':[{'path':key(records),'sha256':sha(records)}],
                  'pieces':[{'path':key(pieces),'sha256':sha(pieces)}]},
        'allow':{'source_groups':list(allow_groups),'target_ids':list(allow_ids)},
        'hold':{'source_groups':list(hold_groups),'target_ids':list(hold_ids)}}


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
