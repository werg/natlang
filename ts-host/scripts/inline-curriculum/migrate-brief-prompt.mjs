#!/usr/bin/env node
/**
 * Give recorded contexts the current system prompt, which ends by asking to be brief.
 *
 *   node scripts/inline-curriculum/migrate-brief-prompt.mjs IN.results.jsonl OUT.results.jsonl
 *
 * A system message that is the prompt as it was before that paragraph is replaced by the current prompt, so a model is
 * trained on the prompt it is served with. Any row can be passed again; each rewritten row is marked in
 * provenance.brief_prompt_migration.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const MIGRATION = 'be-brief/1';
const BRIEF = TOOLS_PROMPT.slice(TOOLS_PROMPT.lastIndexOf('\n\nBe brief.'));
if (!BRIEF.startsWith('\n\nBe brief.')) throw new Error('the current prompt does not end by asking to be brief');
const BEFORE = TOOLS_PROMPT.slice(0, -BRIEF.length) + '\n';

/** The row with the current prompt in its contexts, or the same row when none has the earlier one. */
export function migrateRow(row) {
  let changed = false;
  const trajectory = (row.trajectory ?? []).map(turn => {
    if (!Array.isArray(turn.context) || !turn.context.some(m => m.role === 'system' && m.content === BEFORE)) return turn;
    changed = true;
    return { ...turn, context: turn.context.map(m => m.role === 'system' && m.content === BEFORE ? { ...m, content: TOOLS_PROMPT } : m) };
  });
  if (!changed) return row;
  return { ...row, trajectory, provenance: { ...row.provenance, brief_prompt_migration: MIGRATION } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: migrate-brief-prompt.mjs IN.results.jsonl OUT.results.jsonl');
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0;
  const out = lines.map(line => {
    const row = JSON.parse(line), next = migrateRow(row);
    if (next === row) return line;
    changed++;
    return JSON.stringify(next);
  });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows given the brief prompt -> ${output}\n`);
}
