"""Nondefault arithmetic survives raw identity and durable optimizer updates."""
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('family',['lfm'])
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
