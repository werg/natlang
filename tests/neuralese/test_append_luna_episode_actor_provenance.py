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


def test_static_oracle_binding_preserves_independent_judged_result():
    audit = {"oracle_level": "judged", "exact_oracle_claimed": False,
        "oracle_judge": {"model_id": "gpt-6-sol", "provider": "openai-codex"}}
    raw = {"outcome": {"accepted": True,
        "oracle": {"accepted": True, "level": "judged"}},
        "provenance": {"judge": {"model": "gpt-6-sol", "provider": "openai-codex",
            "transport": "pi-provider"}}}
    assert MODULE._verified_oracle_claim(audit, raw) == {
        "oracle_level": "judged", "exact_oracle_claimed": False,
        "oracle_judge": {"model_id": "gpt-6-sol", "provider": "openai-codex"}}


def test_static_oracle_binding_rejects_mismatched_or_fabricated_judge():
    audit = {"oracle_level": "judged", "exact_oracle_claimed": False,
        "oracle_judge": {"model_id": "gpt-6-sol", "provider": "openai-codex"}}
    raw = {"outcome": {"accepted": True,
        "oracle": {"accepted": True, "level": "judged"}},
        "provenance": {"judge": {"model": "gpt-6-luna", "provider": "openai-codex",
            "transport": "pi-provider"}}}
    with pytest.raises(ValueError, match="matching independent judge provenance"):
        MODULE._verified_oracle_claim(audit, raw)


def test_static_oracle_binding_keeps_exact_claims_explicit():
    audit = {"oracle_level": "exact", "exact_oracle_claimed": True}
    raw = {"outcome": {"accepted": True,
        "oracle": {"accepted": True, "level": "exact"}}}
    assert MODULE._verified_oracle_claim(audit, raw) == {
        "oracle_level": "exact", "exact_oracle_claimed": True}
    with pytest.raises(ValueError, match="exact oracle claim"):
        MODULE._verified_oracle_claim(audit, {**raw, "outcome": {"accepted": True,
            "oracle": {"accepted": True, "level": "judged"}}})


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


def test_big2_recipe_successor_normalizes_actual_published_field_names(tmp_path):
    import hashlib
    refs = {}
    for key in ('successor_source', 'predecessor_source', 'ce_policy', 'recipe',
                'geometry', 'family_map', 'objective_census'):
        path = tmp_path / key
        path.write_text(key)
        refs[key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}
    receipt = {'schema': 'natlang.big2-source-objective-recipe-successor/1',
        'training_admission': False, 'optimizer_exposure': False,
        'successor_rows': 307945, 'current_protection_digest': '8' * 64, **refs}
    normalized = MODULE._normalize_base_receipt(receipt)
    assert normalized['schema'] == 'natlang.native-quality-successor/1'
    assert normalized['source'] == refs['successor_source']
    assert normalized['rows_after'] == 307945
    assert normalized['protection_digest'] == '8' * 64
    assert normalized['status'] == 'source-ce-recipe-geometry-qualified-not-published'


def test_big2_recipe_successor_rejects_changed_actual_successor_binding(tmp_path):
    import hashlib
    refs = {}
    for key in ('successor_source', 'predecessor_source', 'ce_policy', 'recipe',
                'geometry', 'family_map', 'objective_census'):
        path = tmp_path / key
        path.write_text(key)
        refs[key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}
    receipt = {'schema': 'natlang.big2-source-objective-recipe-successor/1',
        'training_admission': False, 'optimizer_exposure': False,
        'successor_rows': 307945, 'current_protection_digest': '8' * 64, **refs}
    (tmp_path / 'successor_source').write_text('tampered')
    with pytest.raises(ValueError, match='binding changed: successor_source'):
        MODULE._normalize_base_receipt(receipt)
