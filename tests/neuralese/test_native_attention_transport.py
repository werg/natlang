"""Nondefault arithmetic survives raw identity and durable optimizer updates."""
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_attention_mode_survives_real_three_step_service_updates_and_relocation(tmp_path,monkeypatch,family):
    import test_native_full_updates as updates
    original=updates.load_native_model
    def load(*args,**kwargs):
        return original(*args,**kwargs,native_attention_precision='tf32x3')
    monkeypatch.setattr(updates,'load_native_model',load)
    engines=[];restore=updates.load_native_engine
    def observe(*args,**kwargs):
        engine=restore(*args,**kwargs);engines.append(engine)
        assert engine.shared_model.native_attention_precision=='tf32x3'
        assert engine.shared_model.port.native_attention_precision=='tf32x3'
        assert engine.model_config['native_attention_precision']=='tf32x3'
        return engine
    monkeypatch.setattr(updates,'load_native_engine',observe)
    # The reused durability regression deliberately corrupts one sidecar at
    # its end. Preserve its exact bytes so the additional mode-tamper check
    # below reaches arithmetic validation rather than that unrelated guard.
    originals={};write_bytes=Path.write_bytes
    def preserve(path,data):
        if data==b'corrupt RNG sidecar':originals[path]=path.read_bytes()
        return write_bytes(path,data)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'write_bytes',preserve)
        updates.test_actual_backbone_update_reload_and_sparse_target_resume(tmp_path,family,'adam')
    for path,data in originals.items():write_bytes(path,data)
    assert len(engines)>=5
    manifests=[tmp_path/'relocated-bundle/checkpoint.json']
    assert manifests
    for path in manifests:
        value=json.loads(path.read_text())
        assert value['model_config']['native_attention_precision']=='tf32x3'
    # A changed arithmetic declaration must not silently override the exact
    # parent while applying an otherwise valid full-model parameter patch.
    value=json.loads(manifests[-1].read_text());value['model_config']['native_attention_precision']='ieee'
    bad=manifests[-1].with_name('bad-attention.json');bad.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='model configuration'):
        restore(str(bad),base=str(tmp_path/'relocated-base'))


def test_async_restoration_checks_saved_attention_mode(tmp_path,monkeypatch):
    import torch
    from test_native_async_eval import fixture
    from natlang_neuralese.model.native_precision import set_native_attention_precision
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA,model_state
    from natlang_neuralese.eval.native_async import seal_request,file_binding
    from natlang_neuralese.eval import task_cadence
    from natlang_neuralese import serve
    base,model,checkpoint,spec,_=fixture(tmp_path)
    config=torch.load(checkpoint,weights_only=False)['model_config']
    set_native_attention_precision(model,'tf32x3');config['native_attention_precision']='tf32x3'
    torch.save({'schema':CHECKPOINT_SCHEMA,'step':7,'model_config':config,**model_state(model)},checkpoint)
    sealed=seal_request(checkpoint,identity=file_binding(checkpoint),step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    directory=Path(sealed['request']);request=json.loads((directory/'request.json').read_text())
    request['dir']=str(directory)
    engine=task_cadence.restore_engine(request)
    try:
        assert engine.model_config['native_attention_precision']=='tf32x3'
        engine.model_config['native_attention_precision']='ieee'
        monkeypatch.setattr(serve,'load_native_engine',lambda *args,**kwargs:engine)
        with pytest.raises(ValueError,match='model contract changed: native_attention_precision'):
            task_cadence.restore_engine(request)
    finally:engine.stop()
