#!/usr/bin/env node
/**
 * Give the offered return_result tool its current description, with the sentence on negative answers.
 *
 *   node scripts/inline-curriculum/migrate-return-description.mjs IN.results.jsonl OUT.results.jsonl
 *
 * The earlier description is replaced by the current one wherever a row carries it. Any row can be passed again; each
 * rewritten row is marked in provenance.return_description_migration.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { RETURN_RESULT_DESCRIPTION, RETURN_RESULT_DESCRIPTION_BEFORE } from '../../dist/native/agent.js';

const MIGRATION = 'negative-answers/1';
const inJson = text => JSON.stringify(text).slice(1, -1);
const escape = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// The current description begins with the earlier one, so the earlier one counts only where the added sentence does not
// follow it: a migrated row is left as it is.
const added = RETURN_RESULT_DESCRIPTION.slice(RETURN_RESULT_DESCRIPTION_BEFORE.length);
const FROM = new RegExp(escape(inJson(RETURN_RESULT_DESCRIPTION_BEFORE)) + `(?!${escape(inJson(added))})`, 'g');
const TO = inJson(RETURN_RESULT_DESCRIPTION);

/** The row's JSON text with the current description, or the same text when it has none to replace. */
export function migrateLine(line) {
  const text = line.replace(FROM, () => TO);
  if (text === line) return line;
  const row = JSON.parse(text);
  row.provenance = { ...row.provenance, return_description_migration: MIGRATION };
  return JSON.stringify(row);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: migrate-return-description.mjs IN.results.jsonl OUT.results.jsonl');
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0;
  const out = lines.map(line => { const next = migrateLine(line); if (next !== line) changed++; return next; });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows given the current return_result description -> ${output}\n`);
}
