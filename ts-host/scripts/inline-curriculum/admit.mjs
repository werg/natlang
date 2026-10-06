#!/usr/bin/env node
// Admit collected curriculum rows: the collector's contract verdict plus the causal checks.
// node scripts/inline-curriculum/admit.mjs RESULTS.jsonl [MORE.jsonl ...] --ledger LEDGER.jsonl [--admitted ADMITTED.jsonl] [--show]
// The ledger holds one admission per row with its concrete rejection reasons; coverage by family,
// slice, domain, mode, inline use, and turn count is printed and written beside the ledger.
import { writeFile } from 'node:fs/promises';
import { jsonlRows } from '../jsonl-stream.mjs';
import { writeAtomic } from '../../dist/teacher/collector.js';
import { parseArgs } from 'node:util';
import { admitRow, callName, coverage, openingLength } from '../../dist/teacher/curriculum.js';

/** Free-text answers need not be perfect (owner 2026-10-06). A span answer below its threshold that no judge has
 * reviewed is admitted when it still overlaps the reference (span F1 >= LENIENT_SPAN), marked as lenient. */
export const LENIENT_SPAN = 0.3;
export function lenientFreeText(row) {
  const oracle = row.outcome?.oracle;
  if (row.outcome?.accepted === true || oracle?.level !== 'span' || !oracle.needs_review || !(oracle.score >= LENIENT_SPAN)) return row;
  // Only the answer check may have failed; every other check (status, effects, files, world) must have passed.
  const reasons = row.outcome.rejection_reasons ?? [];
  if (reasons.some(reason => reason !== 'answer')) return row;
  const pending = (row.outcome.quality_pending ?? []).filter(reason => reason !== 'answer_needs_review');
  return { ...row, outcome: { ...row.outcome, accepted: true, rejection_reasons: [], checks: { ...row.outcome.checks, answer: true },
    quality_pending: pending, oracle: { ...oracle, needs_review: false, accepted: true,
      lenient: `unjudged free text, span F1 ${oracle.score.toFixed(2)} >= ${LENIENT_SPAN}` } } };
}

/**
 * A hinted run trains as if unprompted: its hint paragraph is removed from every message of its trajectory and
 * from the program, so the admitted row shows the behavior without the request for it.
 */
export function stripHint(row) {
  const hint = row.task.program_ir.curriculum.hint;
  if (!hint) return row;
  const text = JSON.stringify(row);
  const stripped = JSON.parse(text.split(JSON.stringify(`\n\n${hint}`).slice(1, -1)).join('').split(JSON.stringify(hint).slice(1, -1)).join(''));
  stripped.task.program_ir.curriculum.hint_stripped = true;
  return stripped;
}

if (import.meta.url === `file://${process.argv[1]}`) await main();

async function main() {
const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  ledger: { type: 'string' }, admitted: { type: 'string' }, show: { type: 'boolean', default: false },
  'require-technique': { type: 'boolean', default: false } } });
if (!positionals.length || !values.ledger) throw new Error('usage: admit.mjs RESULTS.jsonl... --ledger LEDGER.jsonl [--admitted OUT.jsonl] [--require-technique] [--show]');
// With --require-technique, a correct run that skipped the technique its case requires (judged directly) is not
// written to the admitted output: training data meant to teach the technique. The ledger still admits it.
const written = item => item.admitted && !(values['require-technique'] && item.notes?.includes('judged_directly'));

const admissions = [];
const writtenIds = new Set();
async function* admittedRows() {
  for (const path of positionals) {
    for await (let row of jsonlRows(path)) {
      if (!row.task?.program_ir?.curriculum) continue;
      row = lenientFreeText(row);
      const item = admitRow(row);
      // Automatic historical snapshots can overlap explicitly supplied exports.
      // Preserve the admission verdict and record why a duplicate is not emitted.
      const duplicate = written(item) && writtenIds.has(String(row.id));
      if (duplicate) item.notes = [...(item.notes ?? []), 'duplicate_input_trajectory_id_not_written'];
      admissions.push(item);
      if (written(item) && values.admitted && !duplicate) {
        writtenIds.add(String(row.id));
        yield JSON.stringify(stripHint(row)) + '\n';
      }
      const f = item.facts;
      console.log(`${item.admitted ? 'ADMIT ' : 'REJECT'} ${item.program_id}  root turns ${f.rootTurns}, evals ${f.evals}, inline ${f.inlineCalls}, ` +
        `named ${f.namedChildCalls}, edits ${f.edits}${item.reasons.length ? `  — ${item.reasons.join('; ')}` : ''}`);
      if (values.show) {
        const trajectory = row.trajectory ?? [];
        const root = callName(trajectory[0]?.context ?? []);
        for (const turn of trajectory) {
          if (callName(turn.context) !== root) continue;
          const last = turn.context.at(-1);
          if (turn.context.length > openingLength(turn.context)) console.log(`    ← ${String(last.content ?? '').slice(0, 400).replace(/\n/g, '\n      ')}`);
          for (const call of turn.assistant?.calls ?? []) console.log(`    → ${call.tool} ${JSON.stringify(call.arguments).slice(0, 600)}`);
          if (!turn.assistant?.calls?.length && turn.assistant?.content) console.log(`    → reply ${turn.assistant.content.slice(0, 300)}`);
        }
        console.log(`    = ${JSON.stringify(row.outcome?.value)} (${row.outcome?.status}); expected ${JSON.stringify(row.task.program_ir.semantics.expected)}`);
      }
    }
  }
}
if (values.admitted) await writeAtomic(values.admitted, admittedRows());
else for await (const _ of admittedRows()) { /* consume without retaining trajectories */ }
await writeFile(values.ledger, admissions.map(item => JSON.stringify(item)).join('\n') + (admissions.length ? '\n' : ''));
const summary = coverage(admissions);
await writeFile(values.ledger.replace(/\.jsonl$/, '') + '.coverage.json', JSON.stringify(summary, null, 2) + '\n');
console.log(`\n${summary.admitted}/${summary.total} admitted; rejections ${JSON.stringify(summary.rejections)}`);
console.log(`slice shares ${JSON.stringify(summary.shares.slice)}; domain shares ${JSON.stringify(summary.shares.domain)}`);
}
