"""Derivative registration exposes fit entrypoints once and keeps long obligations."""
import json
from pathlib import Path
import pytest
from scripts.neuralese_data.prepare_native_capacity_admission import propose
from scripts.neuralese_data.partition_acquired_native_capacity import partition
from natlang_neuralese.train.native_pending_sources import declaration
from natlang_neuralese.common.hashing import sha256_file_hex
from test_acquired_native_capacity_partition import source


def test_derivative_keeps_original_eligibility_and_exact_pending_set_without_double_source(tmp_path):
    audit,samples,_=source(tmp_path);root=audit.parent
    proposal=root/'full-proposal.json'
    proposal.write_text(json.dumps({'id':'original','include':[path.name for path in root.iterdir() if path.is_file()],
        'documents':{},'build':{'audit':audit.name,'audit_sha256':sha256_file_hex(audit)}}))
    directory=root/'partition';limits={model:{'forward_tokens':5,'kv_slots':5} for model in ('lfm','mellum')}
    receipt=partition(audit,samples,limits,directory);declarations={}
    for model in limits:
        pending=directory/receipt['manifests'][model+':train:remaining-long']
        path=directory/(model+'-pending.json');path.write_text(json.dumps(declaration(root/(model+'-train.jsonl'),pending,
            tokenizer_fingerprint='fp',role_ids=(58,59,60,61))))
        declarations[model]=path
    output=directory/'proposal.json'
    result=propose(proposal,directory/'receipt.json',declarations,cohort_root=root,identifier='derivative',output=output)
    assert result['id']=='derivative' and result['build']['capacity_partition']['source_corpus_id']=='original'
    assert result['build']['capacity_partition']['full_cohort_is_additional_training_source'] is False
    for model in limits:
        assert result['documents'][model]['source_eligible_graphs']==2
        assert result['documents'][model]['fit_train_graphs']==1
        assert result['documents'][model]['pending_long_graphs']==1
        assert result['documents'][model]['earned_graph_visits']==0
        assert result['training_entrypoints'][model]['native_graphs'].endswith('/fit-sources.json')
        assert not result['training_entrypoints'][model]['execution_qualified']
    with pytest.raises(ValueError,match='new immutable corpus ID'):
        propose(proposal,directory/'receipt.json',declarations,cohort_root=root,identifier='original',output=output)
    value=json.loads(declarations['lfm'].read_text());value['sources'][0]['rows'][0]['id']='different-row'
    declarations['lfm'].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='complete unchanged long-row set'):
        propose(proposal,directory/'receipt.json',declarations,cohort_root=root,identifier='derivative',output=output)
