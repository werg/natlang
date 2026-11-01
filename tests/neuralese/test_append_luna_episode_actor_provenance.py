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
