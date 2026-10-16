"""Rolling admission must not treat old marker filenames as current receipts."""
import json

import pytest

from scripts.neuralese_data.admit_completed_search_shards import eligible,convert_one,output_directory,sha


@pytest.mark.parametrize('mutation',['legacy','renderer','aggregates','trajectories'])
def test_old_or_incomplete_renderer_marker_cannot_qualify(mutation):
    receipt={'renderer_code_sha256':'current','aggregate_outputs':{'source':'hash'},
             'trajectory_bindings':{'trajectory':'binding'}}
    assert eligible(receipt,'current')
    if mutation=='legacy':receipt={'records':5000,'trajectories_committed':300}
    if mutation=='renderer':receipt['renderer_code_sha256']='old'
    if mutation=='aggregates':receipt['aggregate_outputs']={}
    if mutation=='trajectories':receipt['trajectory_bindings']={}
    assert not eligible(receipt,'current')


@pytest.mark.parametrize('mutation',['none','source','output','model'])
def test_resume_reuses_only_exact_completed_source_model_and_output(tmp_path,mutation):
    source=tmp_path/'source';source.write_bytes(b'original source')
    output=tmp_path/'output';output.mkdir();graph=output/'graph';graph.write_bytes(b'exact graph')
    receipt={'model_id':'actual-model','source':{'sha256':sha(source)},
        'outputs':[{'path':str(graph),'sha256':sha(graph)}]}
    (output/'receipt.json').write_text(json.dumps(receipt))
    if mutation=='source':source.write_bytes(b'changed source')
    if mutation=='output':graph.write_bytes(b'changed graph')
    model='wrong-model' if mutation=='model' else 'actual-model'
    if mutation=='none':assert convert_one(source,output,'tokenizer',model)==receipt
    else:
        with pytest.raises(ValueError,match='changed'):
            convert_one(source,output,'tokenizer',model)


def test_views_keep_the_existing_distinct_layout(tmp_path):
    assert output_directory(tmp_path,'whole-trajectory','lfm25')==tmp_path/'lfm25'
    assert output_directory(tmp_path,'program-header','lfm25')==tmp_path/'program-header/lfm25'
