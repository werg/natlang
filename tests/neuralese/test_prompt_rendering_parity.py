"""Python prompt_rendering.render_system_prompt equals ts-host renderSystemPrompt (D9 shared boundary).

training/prompt-pieces/parity-fixture.jsonl holds recorded system prompts and the TS program-header rendering
(ts-host/scripts/export-prompt-pieces.mjs --fixture); pieces.json is the exported piece table of the same commit.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.data.prompt_rendering import load_pieces, render_system_prompt  # noqa: E402


def test_python_port_matches_ts_render_system_prompt():
    pieces = load_pieces(ROOT / 'training' / 'prompt-pieces' / 'pieces.json')
    rows = [json.loads(line) for line in (ROOT / 'training' / 'prompt-pieces' / 'parity-fixture.jsonl')
            .read_text(encoding='utf-8').splitlines() if line]
    assert rows
    mismatches = [row['id'] for row in rows
                  if render_system_prompt(row['text'], 'program-header', pieces) != row['program_header']]
    assert not mismatches, mismatches[:5]
    assert all(render_system_prompt(row['text'], 'full', pieces) == row['text'] for row in rows)
