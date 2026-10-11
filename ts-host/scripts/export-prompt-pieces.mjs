#!/usr/bin/env node
// Export ts-host's prompt piece table (promptPieces(): id, scope, text) for the Python training/serving renderer
// (natlang_neuralese/data/prompt_rendering.py), optionally with the historical bank, and optionally render recorded
// system prompts with the TS renderSystemPrompt as a parity fixture.
//
//   node scripts/export-prompt-pieces.mjs OUT.json [--history] [--fixture IN.jsonl FIXTURE_OUT.jsonl]
//
// --history (D9, plans/neuralese/D9_NATIVE_SFT_V13_REPAIR_AUDIT_20261011.md): recorded corpora carry system prompts
// of earlier runtime versions. Every revision of native/prompt.ts and model/text-neuralese-emulation.ts in git is
// loaded (as migrate-system-prompt.mjs promptHistory() does, also in the read_code-renamed form); each exported string
// constant and each derived piece of that revision becomes a historical piece with the scope its current counterpart
// has. Program declarations (depth-limit notice, directory-reducer declaration, Neuralese text-channel emulation,
// Neuralese string boundaries, the edit_code capability line) are 'program'; everything else is 'generic'.
// IN.jsonl rows: {"id": ..., "text": <system prompt>}; FIXTURE_OUT rows add "program_header": renderSystemPrompt(text).
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promptPieces, renderSystemPrompt } from '../dist/native/system-prompts.js';
import { historicalPromptPieces } from './inline-curriculum/migrate-system-prompt.mjs';

const args = process.argv.slice(2);
const out = args[0];
const history = args.includes('--history');
const fixtureAt = args.indexOf('--fixture');
if (!out) { console.error('usage: export-prompt-pieces.mjs OUT.json [--history] [--fixture IN.jsonl OUT.jsonl]'); process.exit(2); }
const REPO = join(dirname(fileURLToPath(import.meta.url)), '../..');
const git = (...a) => execFileSync('git', a, { cwd: REPO, encoding: 'utf8', maxBuffer: 1 << 28 });
const commit = git('log', '-1', '--format=%H', '--', 'ts-host/src/native').trim();
const pieces = promptPieces().map(p => ({ id: p.id, scope: p.scope, text: p.text, historical: false }));

async function historicalPieces() {
  const current = new Set(pieces.map(p => p.text));
  const added = (await historicalPromptPieces()).filter(p => !current.has(p.text));
  return { added, failures: [] };
}

let report = {};
if (history) {
  const { added, failures } = await historicalPieces();
  pieces.push(...added);
  report = { historical: added.length, load_failures: failures };
}
writeFileSync(out, JSON.stringify({ schema: 'natlang.prompt-pieces/1', generated_from_commit: commit, history, pieces, ...report }, null, 1) + '\n');
if (fixtureAt >= 0) {
  const [fixtureIn, fixtureOut] = args.slice(fixtureAt + 1, fixtureAt + 3);
  const rows = readFileSync(fixtureIn, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  writeFileSync(fixtureOut, rows.map(row => JSON.stringify({ ...row, program_header: renderSystemPrompt(row.text, 'program-header', pieces) })).join('\n') + '\n');
}
console.log(JSON.stringify({ pieces: pieces.length, commit, ...report, load_failures: report.load_failures?.length }));
