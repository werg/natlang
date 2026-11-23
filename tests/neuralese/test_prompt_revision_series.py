"""The prompt-revision opt-out (NATLANG_PROMPT_REVISION; plans/semantic-fp/02 R1) is part of the held series: it joins
the resolved model-facing list, so the series key names it and every collector, which receives that list as
NATLANG_MODEL_FACING, serves the same revision (ts-host runtime/model-facing.ts promptRevision)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.rl import held_eval  # noqa: E402


def test_default_has_no_revision(monkeypatch):
    monkeypatch.delenv('NATLANG_PROMPT_REVISION', raising=False)
    monkeypatch.delenv('NATLANG_MODEL_FACING', raising=False)
    assert 'prompt-revision' not in held_eval.model_facing()


def test_revision_joins_the_variants(monkeypatch):
    monkeypatch.delenv('NATLANG_MODEL_FACING', raising=False)
    monkeypatch.setenv('NATLANG_PROMPT_REVISION', '2026-11-22')
    resolved = held_eval.model_facing().split(',')
    assert resolved[-1] == 'prompt-revision-2026-11-22'
    assert 'compact-callable-scope' in resolved   # the adopted variants stay
    monkeypatch.setenv('NATLANG_MODEL_FACING', ','.join(resolved))
    assert held_eval.model_facing().split(',').count('prompt-revision-2026-11-22') == 1
    monkeypatch.setenv('NATLANG_PROMPT_REVISION', 'current')
    monkeypatch.setenv('NATLANG_MODEL_FACING', 'effect-receipt')
    assert held_eval.model_facing() == 'effect-receipt'
