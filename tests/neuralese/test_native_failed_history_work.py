"""Candidate preparation is resumable work, never learner admission."""
import copy
import json
import pytest
from test_native_failed_history_records import fixture
from scripts.neuralese_data import build_native_failed_history_masks as producer
from natlang_neuralese.train.native_failed_history import load_manifest,binding
from natlang_neuralese.common.hashing import sha256_hex


def test_failed_native_custody_resumes_without_rederiving_and_still_rechecks_receipt(tmp_path,monkeypatch):
    tok,_,_,_,source,receipt,receipt_path=fixture(tmp_path)
    work=tmp_path/'work';out=tmp_path/'out'
    broken=copy.deepcopy(receipt);broken['tokenizers']['synthetic']['files']['chat_template.jinja']='a'*64
    receipt_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='template/history'):
        producer.build([source],tokenizer=tok,out=out,native_render_receipts=[receipt_path],work=work)
    assert not out.exists()
    saved=list(work.glob('*.json'));saved=[p for p in saved if p.name!='custody-report.json']
    assert len(saved)==1
    candidate=json.loads(saved[0].read_text())
    assert candidate['admitted']is False
    with pytest.raises(ValueError):load_manifest(binding(saved[0]))
    receipt_path.write_text(json.dumps(receipt))
    def forbidden(*args,**kwargs):raise AssertionError('completed mask derivation repeated')
    monkeypatch.setattr(producer,'derive_context_mask',forbidden)
    # The cache law is bound to the original helper module, not a test replacement.
    forbidden.__module__='natlang_neuralese.train.native_failed_history'
    ref,counts=producer.build([source],tokenizer=tok,out=out,native_render_receipts=[receipt_path],work=work)
    assert counts['affected_rows']==1 and load_manifest(ref)[0]['rows'][0]['source_evidence']is not None


def test_corrupt_candidate_work_fails_before_admission(tmp_path):
    tok,*_,source,receipt,receipt_path=fixture(tmp_path)
    work=tmp_path/'work'
    producer.build([source],tokenizer=tok,out=tmp_path/'first',native_render_receipts=[receipt_path],work=work)
    saved=next(p for p in work.glob('*.json')if p.name!='custody-report.json')
    value=json.loads(saved.read_text());value['payload']['candidates'][0]['record']['zero_spans']=[[0,1000000]]
    saved.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='work binding'):
        producer.build([source],tokenizer=tok,out=tmp_path/'second',native_render_receipts=[receipt_path],work=work)
    assert not(tmp_path/'second').exists()


def test_capture_census_reports_every_requested_failed_row_without_admitting(tmp_path,monkeypatch):
    tok,_,_,graph,source,_,_=fixture(tmp_path)
    raw=tmp_path/'capture.jsonl';lines=[json.dumps({'id':name})for name in('one','two')]
    raw.write_text('\n'.join(lines)+'\n');raw_binding=binding(raw)
    rows=[]
    for index,name in enumerate(('one','two')):
        row=copy.deepcopy(graph);row['id']=name
        row['conversion']['teacher_source']={'source_artifact':str(raw),
            'source_artifact_sha256':raw_binding['sha256'],'source_row_id':name,
            'source_row_sha256':sha256_hex(lines[index].encode()),'teacher_turn_index':0}
        rows.append(row)
    source.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    visited=[]
    def reject(graph,raw,line_sha256,**kwargs):
        visited.append(raw['id']);raise ValueError('synthetic exact-custody refusal')
    monkeypatch.setattr(producer,'capture_evidence',reject)
    work=tmp_path/'work'
    with pytest.raises(ValueError,match='2 errors; no manifest'):
        producer.build([source],tokenizer=tok,out=tmp_path/'out',work=work)
    report=json.loads((work/'custody-report.json').read_text())
    assert visited==['one','two']and report['admitted']is False
    assert {error['id']for error in report['errors']}=={'one','two'}
    assert all(error['source_sha256']==binding(source)['sha256']for error in report['errors'])
    assert not(tmp_path/'out').exists()


def test_checked_prior_module_reuses_v1_work_only_for_identical_callable_law(tmp_path,monkeypatch):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    from natlang_neuralese.train import native_failed_history as helper
    from pathlib import Path
    tok,*_,source,receipt,receipt_path=fixture(tmp_path)
    work=tmp_path/'work'
    producer.build([source],tokenizer=tok,out=tmp_path/'first',native_render_receipts=[receipt_path],work=work)
    cache=next(p for p in work.glob('*.json')if p.name!='custody-report.json')
    value=json.loads(cache.read_text());payload=value['payload'];old=tmp_path/'old-helper.py'
    old.write_text(Path(helper.__file__).read_text())
    payload['derivation'].pop('callable_law_sha256');payload['derivation'].update(version=1,mask_module_sha256=binding(old)['sha256'])
    payload['key']=digest({'source_sha256':binding(source)['sha256'],**payload['derivation']})
    value['payload_sha256']=digest(payload);cache.unlink();(work/(payload['key']+'.json')).write_text(json.dumps(value))
    def forbidden(*args,**kwargs):raise AssertionError('old completed derivation repeated')
    forbidden.__module__=helper.__name__;monkeypatch.setattr(producer,'derive_context_mask',forbidden)
    producer.build([source],tokenizer=tok,out=tmp_path/'resumed',native_render_receipts=[receipt_path],work=work,prior_derivation_modules=[old])
    old.write_text(old.read_text().replace("node=row['nodes'][0];ids=node['token_ids']","node=row['nodes'][0];ids=node['token_ids'][:-1]"))
    with pytest.raises(ValueError,match='actual mask callable law'):
        producer.build([source],tokenizer=tok,out=tmp_path/'refused',native_render_receipts=[receipt_path],work=work,prior_derivation_modules=[old])


def test_custody_only_module_edit_keeps_v2_work_but_mask_edit_invalidates(tmp_path,monkeypatch):
    from pathlib import Path
    from natlang_neuralese.train import native_failed_history as helper
    tok,*_,source,receipt,receipt_path=fixture(tmp_path)
    work=tmp_path/'work'
    producer.build([source],tokenizer=tok,out=tmp_path/'first',native_render_receipts=[receipt_path],work=work)
    original=Path(helper.__file__).read_text();changed=tmp_path/'updated-helper.py'
    changed.write_text(original.replace('def validate_source_evidence(row,evidence,*,target_render_binding=None):',
        'def validate_source_evidence(row,evidence,*,target_render_binding=None):\n    assert evidence is not None'))
    assert changed.read_text()!=original
    monkeypatch.setattr(helper,'__file__',str(changed))
    def forbidden(*args,**kwargs):raise AssertionError('derivation executed')
    monkeypatch.setattr(producer,'derive_context_mask',forbidden)
    # Actual module bytes changed, but custody-only code is not a derivation law.
    producer.build([source],tokenizer=tok,out=tmp_path/'resumed',native_render_receipts=[receipt_path],work=work)
    changed.write_text(changed.read_text().replace("node=row['nodes'][0];ids=node['token_ids']","node=row['nodes'][0];ids=node['token_ids'][:-1]"))
    with pytest.raises(AssertionError,match='derivation executed'):
        producer.build([source],tokenizer=tok,out=tmp_path/'new-law',native_render_receipts=[receipt_path],work=work)
