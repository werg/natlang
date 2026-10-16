"""Whole graphs/splits survive capacity selection with exact original row bytes."""
import json
from pathlib import Path
import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.data.native_graph import parse_graph
from scripts.neuralese_data.partition_acquired_native_capacity import ordinary_geometry,partition


def source(tmp_path):
    root=tmp_path/'source';root.mkdir();proofs=[];conversions=[];original={}
    for model in ('lfm','mellum'):
        outputs=[]
        for split,widths in [('train',[4,6]),('test',[9])]:
            path=root/(model+'-'+split+'.jsonl');lines=[]
            for index,width in enumerate(widths):
                row={'schema':'natlang.neuralese-native-graph/1','id':model+split+str(index),'split':split,
                    'source_groups':[split+'-task'],'tokenizer_fingerprint':'fp',
                    'nodes':[{'name':'own','token_ids':list(range(width)),'target_weights':[0]+[1]*(width-1)}]}
                line=json.dumps(row,indent=None,separators=(', ', ': '))+'\n';lines.append(line)
                proof={'model':model,'id':row['id'],'split':split,'tokens':width,'graph_sha256':canonical_json_sha256_hex(row)}
                example=parse_graph(row,tokenizer_fingerprint='fp')
                assert ordinary_geometry(row,proof)=={'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)}
                proofs.append(proof);original[model,row['id']]=line
            path.write_text(''.join(lines));outputs.append({'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size,
                'rows':len(widths),'split':split})
        conversions.append({'model':model,'outputs':outputs})
    proof=root/'row-proof.jsonl';proof.write_text(''.join(json.dumps(row)+'\n' for row in proofs))
    audit=root/'audit.json';audit.write_text(json.dumps({'schema':'natlang.acquired-session-admission-audit/1',
        'status':'passed','by_model':{'lfm':{},'mellum':{}},'row_proof_sha256':sha256_file_hex(proof),'conversions':conversions}))
    sample=root/'samples.json';sample.write_text(json.dumps({'passed':True,'cohort_audit':{'path':str(audit),'sha256':sha256_file_hex(audit)}}))
    return audit,sample,original


def test_partition_preserves_exact_rows_all_held_views_pending_and_idempotent_resume(tmp_path):
    audit,samples,original=source(tmp_path);limits={'lfm':{'forward_tokens':5,'kv_slots':5},'mellum':{'forward_tokens':7,'kv_slots':7}}
    out=tmp_path/'partition';receipt=partition(audit,samples,limits,out)
    assert receipt['graphs']==6 and receipt['counts']['lfm']['train:fit']['rows']==1
    assert receipt['counts']['mellum']['train:fit']['rows']==2
    assert not receipt['execution_qualified'] and not receipt['full_recipe_completed']
    seen={}
    for path in (out/'files').glob('*/*/*.jsonl.zst'):
        with open_jsonl_text(path) as stream:
            for line in stream:
                row=json.loads(line);key=(path.parent.parent.name,row['id'])
                assert key not in seen and line==original[key]
                seen[key]=line
    assert seen==original
    pending=json.loads((out/receipt['pending_source_declaration']).read_text())
    assert [(r['model'],r['id'],r['required_visits']) for r in pending['rows']]==[('lfm','lfmtrain1',1)]
    assert all('completed_visits' not in r for r in pending['rows']),'declaration is not parallel training progress'
    assert partition(audit,samples,limits,out)==receipt
    with pytest.raises(ValueError,match='different source/envelope'):partition(audit,samples,{m:{'forward_tokens':10,'kv_slots':10} for m in limits},out)
    target=next((out/'files').glob('*/*/fit.jsonl.zst'));target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='immutable'):partition(audit,samples,limits,out)


def test_uncommitted_partition_is_receipted_before_retry_cleanup_and_source_change_is_rejected(tmp_path):
    audit,samples,_=source(tmp_path);out=tmp_path/'out';pending=out/'pending';pending.mkdir(parents=True)
    (pending/'partial').write_text('interrupted compression')
    limits={m:{'forward_tokens':5,'kv_slots':5} for m in ('lfm','mellum')}
    partition(audit,samples,limits,out)
    cleanup=[json.loads(line) for line in (out/'cleanup-receipt.jsonl').read_text().splitlines()]
    assert cleanup[0]['directory']=='pending' and cleanup[0]['files'][0]['sha256']
    actual=json.loads(audit.read_text());path=Path(actual['conversions'][0]['outputs'][0]['path'])
    path.write_text(path.read_text().replace('"token_ids": [0','"token_ids": [99',1))
    with pytest.raises(ValueError,match='immutable'):partition(audit,samples,limits,tmp_path/'new')


def test_logical_cohort_symlink_preserves_exact_physical_audit_binding(tmp_path):
    audit,samples,_=source(tmp_path)
    alias=tmp_path/'logical-cohort';alias.symlink_to(audit.parent,target_is_directory=True)
    limits={m:{'forward_tokens':5,'kv_slots':5} for m in ('lfm','mellum')}
    receipt=partition(alias/audit.name,alias/samples.name,limits,alias/'capacity')
    assert receipt['binding']['audit']['path']==str(audit.resolve())
    assert receipt['graphs']==6
    assert partition(audit,samples,limits,audit.parent/'capacity')==receipt
    alternate=json.loads(samples.read_text());alternate['cohort_audit']['path']=str(alias/audit.name)
    samples.write_text(json.dumps(alternate))
    assert partition(audit,samples,limits,audit.parent/'capacity-through-alias')['graphs']==6
