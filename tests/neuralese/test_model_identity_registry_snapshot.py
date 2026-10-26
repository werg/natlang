import json
from pathlib import Path

from scripts.coreweave.teacher_internal_conversion import _registered_model_identity
from scripts.coreweave.teacher_tranche_admission import file_binding, read_bound


def test_model_identity_survives_subsequent_registry_registration(tmp_path):
    root=tmp_path/'repo';package=root/'runs/corpus/lfm25';(package/'train').mkdir(parents=True)
    graph=package/'train/native-graphs.jsonl.zst';graph.write_bytes(b'exact registered graph')
    receipt=package/'receipt.json';receipt.write_text(json.dumps({'status':'passed','model_key':'lfm25','model_id':'lfm2:base'}))
    training=root/'training';(training/'corpus-manifests').mkdir(parents=True)
    manifest=training/'corpus-manifests/corpus.json'
    entry={'id':'corpus','path':'runs/corpus','training_admission':True}
    manifest.write_text(json.dumps({'id':'corpus','path':entry['path'],'files':[
        {**file_binding(receipt),'path':'lfm25/receipt.json'},
        {**file_binding(graph),'path':'lfm25/train/native-graphs.jsonl.zst'}]}))
    registry=training/'neuralese_corpora.json';registry.write_text(json.dumps({'corpora':[entry]}))
    output=tmp_path/'identity';output.mkdir()
    proof_path=_registered_model_identity(root,'lfm25',[graph],output)
    proof=json.loads(proof_path.read_text());snapshot=read_bound(proof['registry'])
    assert Path(proof['registry']['path'])!=registry
    registry.write_text(json.dumps({'corpora':[entry,{'id':'new-corpus','training_admission':True}]}))
    assert read_bound(proof['registry'])==snapshot=={'corpora':[entry]}
    assert proof['sources'][0]['corpus_id']=='corpus'
