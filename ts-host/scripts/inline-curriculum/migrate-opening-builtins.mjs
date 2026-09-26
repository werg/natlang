#!/usr/bin/env node
/**
 * Add the line naming eval's built-ins to every call opening in recorded contexts, as the agent now writes it.
 *
 *   node scripts/inline-curriculum/migrate-opening-builtins.mjs IN.results.jsonl OUT.results.jsonl
 *
 * A call's opening is the user message that begins "You are inside this call:"; the line ends it. Any row can be
 * passed again; each rewritten row is marked in provenance.opening_builtins_migration.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { BUILT_INS_LINE } from '../../dist/native/agent.js';

const MIGRATION = 'builtins-line/1';
const isOpening = message => message.role === 'user' && typeof message.content === 'string' &&
  message.content.startsWith('You are inside this call:');

/** The row with the line added to each opening that lacks it, or the same row when none does. */
export function migrateRow(row) {
  let changed = false;
  const trajectory = (row.trajectory ?? []).map(turn => {
    if (!Array.isArray(turn.context) || !turn.context.some(m => isOpening(m) && !m.content.includes(BUILT_INS_LINE))) return turn;
    changed = true;
    return { ...turn, context: turn.context.map(m => isOpening(m) && !m.content.includes(BUILT_INS_LINE) ?
      { ...m, content: `${m.content}\n\n${BUILT_INS_LINE}` } : m) };
  });
  if (!changed) return row;
  return { ...row, trajectory, provenance: { ...row.provenance, opening_builtins_migration: MIGRATION } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: migrate-opening-builtins.mjs IN.results.jsonl OUT.results.jsonl');
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0;
  const out = lines.map(line => {
    const row = JSON.parse(line), next = migrateRow(row);
    if (next === row) return line;
    changed++;
    return JSON.stringify(next);
  });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows given the built-ins line -> ${output}\n`);
}
