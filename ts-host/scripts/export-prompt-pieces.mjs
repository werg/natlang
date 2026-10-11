#!/usr/bin/env node
// Export ts-host's prompt piece table (promptPieces(): id, scope, text) for the Python training/serving renderer
// (natlang_neuralese/data/prompt_rendering.py), and optionally render recorded system prompts with the TS
// renderSystemPrompt as a parity fixture.
// Usage: node scripts/export-prompt-pieces.mjs OUT.json [--fixture IN.jsonl FIXTURE_OUT.jsonl]
//   IN.jsonl rows: {"id": ..., "text": <system prompt>}; FIXTURE_OUT rows add "program_header": renderSystemPrompt(text).
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { promptPieces, renderSystemPrompt } from '../dist/native/system-prompts.js';

const [out, flag, fixtureIn, fixtureOut] = process.argv.slice(2);
if (!out) { console.error('usage: export-prompt-pieces.mjs OUT.json [--fixture IN.jsonl OUT.jsonl]'); process.exit(2); }
const commit = execFileSync('git', ['log', '-1', '--format=%H', '--', 'src/native'], { encoding: 'utf8' }).trim();
const pieces = promptPieces().map(p => ({ id: p.id, scope: p.scope, text: p.text, historical: false }));
writeFileSync(out, JSON.stringify({ schema: 'natlang.prompt-pieces/1', generated_from_commit: commit, pieces }, null, 1) + '\n');
if (flag === '--fixture') {
  const rows = readFileSync(fixtureIn, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  writeFileSync(fixtureOut, rows.map(row => JSON.stringify({ ...row, program_header: renderSystemPrompt(row.text, 'program-header') })).join('\n') + '\n');
}
console.log(JSON.stringify({ pieces: pieces.length, commit }));
