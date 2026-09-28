#!/usr/bin/env node
/** Audit raw job results once (not range exports), retaining compact evidence for every rejected row. */
import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { admitRow, obsoleteOutcomes } from '../dist/teacher/curriculum.js';
import { quarantineReason, retiredFamily } from '../dist/teacher/curriculum-policy.js';

const [output, ...roots] = process.argv.slice(2);
if (!output || !roots.length) throw new Error('usage: audit-rejections.mjs OUTPUT_DIR JOBS_OR_RUNS_DIR...');
await mkdir(output, { recursive: true });
async function* files(path) {
  for (const entry of (await readdir(path, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.isDirectory() && !/^(runtime|node_modules|\.git)/.test(entry.name)) yield* files(join(path, entry.name));
    else if (entry.isFile() && /\.(result|error)\.json$/.test(entry.name)) yield join(path, entry.name);
  }
}
const bump = (counts, key) => counts[key] = (counts[key] ?? 0) + 1;
const snippet = value => JSON.stringify(value)?.slice(0, 1000);
const summary = { results: 0, task_rejected: 0, admission_rejected: 0, errors: 0, by_batch: {},
  reasons: {}, diagnostics: {}, error_messages: {}, obsolete: {}, families: {},
  review_categories: {},
  admitted_delegation: { direct: 0, inline: 0, named_only: 0 } };
const rejected = [], errors = [], seen = new Set();
for (const root of roots) for await (const file of files(root)) {
  if (seen.has(file)) continue;
  seen.add(file);
  let row;
  try { row = JSON.parse(await readFile(file, 'utf8')); }
  catch (error) { errors.push({ file, error: String(error), category: 'unreadable_artifact' }); continue; }
  if (file.endsWith('.error.json')) {
    // Export errors left from older collectors do not override a saved result for the same index.
    const batchFiles = await readdir(dirname(file));
    if (batchFiles.some(name => name.startsWith(file.split('/').pop().slice(0, 6) + '-') && name.endsWith('.result.json'))) continue;
    summary.errors++;
    const error = String(row.error);
    const category = /Cannot find package|Cannot find module/i.test(error) ? 'missing_dependency' :
      /Context size|context.*exceed/i.test(error) ? 'context_limit' : /429|rate.limit|RESOURCE_EXHAUSTED/i.test(error) ?
      'rate_limit' : /request budget.*exceeded|budget exhausted/i.test(error) ? 'request_budget' :
      /HeadersTimeout|TimeoutError/i.test(error) ? 'transport_timeout' : /malformed tool arguments/i.test(error) ?
      'malformed_tool_arguments' : /abort|cancel/i.test(error) ? 'cancelled' : 'other';
    bump(summary.error_messages, category);
    errors.push({ file, ...row, category }); continue;
  }
  if (!row.task?.program_ir || !row.outcome) { errors.push({ file, category: 'unsupported_result_shape' }); continue; }
  summary.results++;
  const batch = dirname(file), program = row.task.program_ir, c = program.curriculum;
  const admission = c?.decisive && c?.reference ? admitRow(row) : undefined;
  const taskRejected = row.outcome.accepted !== true, admissionRejected = admission?.admitted === false;
  const counts = summary.by_batch[batch] ??= { total: 0, task_rejected: 0, admission_rejected: 0 };
  counts.total++;
  if (taskRejected) { counts.task_rejected++; summary.task_rejected++; }
  if (admissionRejected) { counts.admission_rejected++; summary.admission_rejected++; }
  if (admission?.admitted) bump(summary.admitted_delegation,
    admission.facts.inlineCalls ? 'inline' : admission.facts.namedChildCalls ? 'named_only' : 'direct');
  if (!taskRejected && !admissionRejected) continue;
  const actions = row.outcome.action_ledger?.filter(event => event.kind === 'action') ?? [];
  const failures = actions.filter(event => ['error', 'rejected', 'failed'].includes(event.outcome));
  const diagnoses = [...new Set(failures.flatMap(event => event.diagnostics ?? []))];
  const obsolete = obsoleteOutcomes(row.trajectory ?? []);
  const reasons = [...new Set([...(admission?.reasons ?? []), ...(row.outcome.rejection_reasons ?? []),
    ...(taskRejected ? ['task_contract'] : [])])];
  // These are review routes, not inferred root causes. Older rows may not carry an answer oracle/check breakdown.
  const reviewCategory = retiredFamily(program) ? 'retired_exercise' : quarantineReason(program) ?
    'unverified_task_contract' : row.outcome.quality_pending?.length ? 'pending_independent_review' : obsolete.length ? 'obsolete_runtime_history' :
    row.outcome.files_check?.accepted === false ? 'file_contract_review' : row.outcome.oracle?.accepted === false ?
    'answer_review' : taskRejected ? 'execution_contract_review' : 'evidence_or_technique_review';
  bump(summary.review_categories, reviewCategory);
  for (const reason of reasons) bump(summary.reasons, reason.split(':')[0]);
  for (const diagnosis of diagnoses) bump(summary.diagnostics, diagnosis);
  for (const item of obsolete) bump(summary.obsolete, item);
  bump(summary.families, c?.family ?? 'non_curriculum');
  const toolCounts = {};
  for (const turn of row.trajectory ?? []) for (const call of turn.assistant?.calls ?? []) bump(toolCounts, call.tool);
  rejected.push({ file, id: program.id, family: c?.family, retired: retiredFamily(program), model: row.provenance?.model,
    task_rejected: taskRejected, admission_rejected: admissionRejected, review_category: reviewCategory, reasons, diagnostics: diagnoses, obsolete,
    status: row.outcome.status, detail: String(row.outcome.detail ?? '').slice(0, 800),
    actual: snippet(row.outcome.value), expected: snippet(program.semantics.expected), oracle: row.outcome.oracle,
    files_check: row.outcome.files_check, turns: row.trajectory?.length ?? 0, tool_counts: toolCounts,
    facts: admission?.facts,
    failures: failures.map(event => ({ tool: event.name, outcome: event.outcome, diagnostics: event.diagnostics,
      code: snippet(event.arguments), result: String(event.result_text ?? '').slice(0, 1200) })) });
}
await writeFile(join(output, 'rejections.jsonl'), rejected.map(row => JSON.stringify(row) + '\n').join(''));
await writeFile(join(output, 'errors.jsonl'), errors.map(row => JSON.stringify(row) + '\n').join(''));
await writeFile(join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
