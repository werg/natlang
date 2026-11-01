import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/neuralese_data/append_luna_episode_to_native_successor.py"
SPEC = importlib.util.spec_from_file_location("append_luna_episode", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_teacher_actor_binding_uses_raw_luna_provenance_not_encoder():
    raw = {"provenance": {
        "model": "gpt-6-luna", "provider": "openai-codex",
        "transport": "pi-provider", "collection_role": "teacher",
        "pi_options": {"reasoningEffort": "low"},
    }}
    raw_binding = {"path": "/immutable/raw.jsonl", "sha256": "a" * 64, "bytes": 10}
    assert MODULE._teacher_actor_binding(raw, raw_binding) == {
        "model": "gpt-6-luna", "provider": "openai-codex",
        "transport": "pi-provider", "collection_role": "teacher",
        "reasoning_effort": "low", "raw_result": raw_binding,
    }


@pytest.mark.parametrize("provenance", [
    {"model": "LFM3", "provider": "native", "transport": "native-state", "collection_role": "teacher"},
    {"model": "gpt-6-luna", "provider": "openai-codex", "transport": "pi-provider", "collection_role": "student"},
])
def test_teacher_actor_binding_rejects_encoder_or_nonteacher(provenance):
    with pytest.raises(ValueError, match="authenticated Luna teacher provenance"):
        MODULE._teacher_actor_binding({"provenance": provenance}, {"sha256": "b" * 64})


def test_static_receipt_rejects_changed_graph_before_loading_runtime(tmp_path):
    import json,hashlib
    graph=tmp_path/'graph.jsonl';graph.write_text('{}\n')
    audit=tmp_path/'audit.json';audit.write_text('{}')
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'schema':'natlang.luna-static-episode-build/1','training_admission':False,'graph':{'path':str(graph),'sha256':'0'*64},'audit':{'path':str(audit),'sha256':hashlib.sha256(audit.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError,match='bound artifact changed'):
        MODULE._verify_episode(receipt,object(),'1'*64)


def test_big2_predecessor_requires_every_sealed_binding(tmp_path):
    import hashlib
    keys=('source_predecessor','source_successor','ce_policy','recipe','geometry','objective_census','family_map')
    refs={}
    for key in keys:
        path=tmp_path/key;path.write_text(key)
        refs[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt={'schema':'natlang.big2-sft-source-append/1','training_admission':False,'optimizer_exposure':False,'rows':298507,**refs}
    assert MODULE._normalize_base_receipt(receipt)['source']==refs['source_successor']
    (tmp_path/'objective_census').write_text('changed')
    with pytest.raises(ValueError,match='binding changed'):
        MODULE._normalize_base_receipt(receipt)
