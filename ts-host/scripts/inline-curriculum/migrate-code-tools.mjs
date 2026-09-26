#!/usr/bin/env node
/**
 * Rewrite collected result rows from the tools read_function, edit_function and diff_functions to read_code,
 * edit_code and diff_code.
 *
 *   node scripts/inline-curriculum/migrate-code-tools.mjs IN.results.jsonl OUT.results.jsonl
 *
 * The names are identifiers, so each is replaced as a whole word throughout the row's JSON: offered tool schemas,
 * calls (model responses, assistant records, replayed contexts, the action ledger), tool results, prompt text and
 * the model's reasoning, which should name the tools it calls. The offered tools' descriptions become the current
 * ones. Any row can be passed again; each rewritten row is marked in provenance.code_tools_migration. A row's
 * program IR (reference solutions name the tools too) is rewritten with it, and its program_ir_sha256 follows, so
 * the row still matches its program in a shard migrated the same way:
 *
 *   node scripts/inline-curriculum/migrate-code-tools.mjs --ir IN.ir.jsonl OUT.ir.jsonl
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { READ_CODE_DESCRIPTION, EDIT_CODE_DESCRIPTION, DIFF_CODE_DESCRIPTION } from '../../dist/native/agent.js';
import { recordDigest } from '../../dist/teacher/collector.js';

const MIGRATION = 'read_code/1';
const NAMES = [[/\bread_function\b/g, 'read_code'], [/\bedit_function\b/g, 'edit_code'], [/\bdiff_functions\b/g, 'diff_code']];
// The descriptions offered before the rename, as they appear inside a row's JSON.
const inJson = text => JSON.stringify(text).slice(1, -1);
const DESCRIPTIONS = [
  ['Read the source of an imported function by its listed name, or the declaration of an external service or an importable package ("pkg" lists its exports, "pkg.name" shows one).', READ_CODE_DESCRIPTION],
  ['Replace one exact or uniquely fuzzy span in an imported function source. The function is validated before the edit becomes live.', EDIT_CODE_DESCRIPTION],
  ['Inspect source changes made to imported functions in this call.', DIFF_CODE_DESCRIPTION],
].map(([from, to]) => [inJson(from), inJson(to)]);

const rewrite = line => DESCRIPTIONS.reduce((out, [from, to]) => out.split(from).join(to),
  NAMES.reduce((out, [from, to]) => out.replace(from, to), line));

/** A program IR record's JSON text in the new surface, or the same text when it has nothing to rewrite. */
export const migrateIrLine = rewrite;

/** A result row's JSON text in the new surface, or the same text when it has nothing to rewrite. */
export function migrateLine(line) {
  const text = rewrite(line);
  if (text === line) return line;
  const before = JSON.parse(line), row = JSON.parse(text);
  // The digest names the program the row ran; it follows the program's rewrite, where it named that program before.
  const program = before.task?.program_ir;
  const digest = program && recordDigest(program) === before.provenance?.program_ir_sha256 ?
    { program_ir_sha256: recordDigest(row.task.program_ir) } : {};
  row.provenance = { ...row.provenance, ...digest, code_tools_migration: MIGRATION };
  return JSON.stringify(row);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2), ir = args[0] === '--ir';
  const [input, output] = ir ? args.slice(1) : args;
  if (!input || !output) throw new Error('usage: migrate-code-tools.mjs [--ir] IN.jsonl OUT.jsonl');
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0;
  const out = lines.map(line => { const next = (ir ? migrateIrLine : migrateLine)(line); if (next !== line) changed++; return next; });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows rewritten -> ${output}\n`);
}
