"""CPU fixture tests exercise the diagnostic plumbing, not CUDA qualification."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/'scripts/neuralese_data/diagnose_native_decision_replay.py'
spec=importlib.util.spec_from_file_location('decision_diagnostic',path)
diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)


def test_real_cpu_engine_collects_and_replays_producer_import_and_decision(tmp_path,monkeypatch):
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    monkeypatch.setattr(diagnostic,'EXPECTED',engine.native_identity['checkpoint_sha256'])
    engine.model_config['native_shared_execution']='ffn-stable-v1'
    selected=iter([7,2])
    def choose(seq):
        token=next(selected)
        return token,float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[token])
    monkeypatch.setattr(engine,'_native_token',choose)
    engine.start()
    try:
        out=tmp_path/'diagnostic';out.mkdir()
        rollout,observed=diagnostic.collect(engine,options=['t7','t8 t9'],out=out)
        result=diagnostic.probe(engine,rollout,out)
        assert result['sampled_tokens']>0 and result['decisions']==1 and result['option_counts']==[2]
        assert observed['write']['id'].startswith('nzs1_')
        from natlang_neuralese.serve.store import decode_native_state
        block,ident=decode_native_state((out/'producer-state.safetensors').read_bytes())
        assert ident==observed['write']['id']
    finally:engine.stop()
