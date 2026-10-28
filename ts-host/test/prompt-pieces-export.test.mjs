import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { promptPieces, renderSystemPrompt } from '../dist/native/system-prompts.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

// The Python training and serving renderer (natlang_neuralese/data/prompt_rendering.py) drops generic runtime
// instructions in program-header mode by matching them against training/prompt-pieces/pieces.json. A piece the
// runtime emits that the table lacks stays in every rendered training row while ts-host drops it at inference.
// Regenerate with: node scripts/export-prompt-pieces.mjs ../training/prompt-pieces/pieces.json --history
const table = JSON.parse(readFileSync(new URL('../../training/prompt-pieces/pieces.json', import.meta.url), 'utf8'));
const exported = table.pieces.filter(piece => ['generic', 'program'].includes(piece.scope));

test('every current runtime prompt piece is in the exported table with its scope', () => {
  const byText = new Map(exported.map(piece => [piece.text, piece.scope]));
  const stale = promptPieces().filter(piece => byText.get(piece.text) !== piece.scope)
    .map(piece => `${piece.id} (${piece.scope}): ${piece.text.slice(0, 100)}`);
  assert.deepEqual(stale, [], 'training/prompt-pieces/pieces.json is stale; regenerate it (see this file)');
});

test('the exported table renders the runtime prompt as the runtime does', () => {
  assert.equal(renderSystemPrompt(TOOLS_PROMPT, 'program-header', exported), renderSystemPrompt(TOOLS_PROMPT, 'program-header'));
});
