#!/usr/bin/env node
/**
 * Advisory failure explanation (plans/FAILURE_EXPLANATION_PROGRAM.md, items P5 and P11). Runs after a crisp tool or a
 * gate has finished; writes a separate file labelled advisory with the explainer's hash. Nothing the pipeline acts on
 * reads that file.
 *
 *   node scripts/explain-advisory.mjs failures --cards CARDS.json [--out FILE] --server URL --model NAME
 *       CARDS.json comes from `scripts/audit_student_projection_failures.py --explain-input`.
 *   node scripts/explain-advisory.mjs rejections --ledger LEDGER.jsonl [--out FILE] --server URL --model NAME
 *       LEDGER.jsonl: one JSON object per line with `reasons` (list) or `reason`, optional `id` and `family`.
 *   node scripts/explain-advisory.mjs gate --facts FACTS.json --report REPORT.json [--out FILE] --server URL --model NAME
 *       FACTS.json comes from `scripts/explain_gate.py`.
 *
 * `--dry-run` prints what would be sent and calls no model. Needs a served model (--server is an OpenAI-compatible
 * endpoint); nothing here loads one. Output files are created fresh and never overwritten.
 */
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { classifyAdmissionReason, RULES } from './admission-dispositions.mjs';
import { advisoryFile, explainerIdentity, refuseExisting, writeAdvisory } from './advisory-file.mjs';

const root = fileURLToPath(new URL('../../applications/failure-explainer/', import.meta.url));
const dist = new URL('../../applications/dist/failure-explainer/index.js', import.meta.url).href;
const sources = () => Object.fromEntries(['explainFailure', 'triageRejections', 'explainGateFailure']
  .map(name => [name, readFileSync(join(root, `${name}.nl`), 'utf8')]));

/** Reasons of a ledger, one row per (item, reason). */
export function ledgerReasons(text) {
  const rows = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const item = JSON.parse(line);
    const reasons = Array.isArray(item.reasons) ? item.reasons : item.reason !== undefined ? [item.reason] : [];
    for (const reason of reasons) rows.push({ reason: String(reason), id: item.id, family: item.family ?? item.task?.program_ir?.family });
  }
  return rows;
}

export async function main(argv, { createRuntime } = {}) {
  const [command, ...rest] = argv;
  const { values } = parseArgs({ args: rest, options: { cards: { type: 'string' }, ledger: { type: 'string' }, facts: { type: 'string' },
    report: { type: 'string' }, out: { type: 'string' }, server: { type: 'string' }, model: { type: 'string' }, 'dry-run': { type: 'boolean' } } });
  const app = await import(dist);
  const identity = explainerIdentity(sources(), values.model ?? 'unspecified');
  const runtime = async () => {
    if (createRuntime) return createRuntime();
    if (!values.server || !values.model) throw new Error('--server and --model are required (or --dry-run)');
    const { createNatlangRuntime, openAICompatibleModelTurn } = await import(new URL('../dist/index.js', import.meta.url).href);
    return createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: values.server, model: values.model }), seed: { mode: 'backend' } });
  };
  const finish = (name, defaultOut, inputs, items) => {
    const out = values.out ?? defaultOut;
    writeAdvisory(out, advisoryFile(name, identity, inputs, items));
    console.log(JSON.stringify({ advisory: out, explainer: identity.explainer }));
  };
  if (command === 'failures') {
    const input = JSON.parse(readFileSync(values.cards, 'utf8'));
    if (values['dry-run']) { console.log(JSON.stringify({ cards: input.cards.length, tags: input.tags, explainer: identity.explainer })); return 0; }
    refuseExisting(values.out ?? `${input.report}.advisory.json`);
    const rt = await runtime();
    try {
      const results = await app.explainFailures(rt, input.cards, input.tags);
      finish('explainFailure', values.out ?? `${input.report}.advisory.json`, input, results);
    } finally { rt.close?.(); }
    return 0;
  }
  if (command === 'rejections') {
    const rows = ledgerReasons(readFileSync(values.ledger, 'utf8'));
    const classified = [...new Set(rows.map(row => row.reason))].map(reason => ({ reason, category: classifyAdmissionReason(reason).category }));
    const bucket = app.unclassifiedBucket(rows, reason => classifyAdmissionReason(reason).category);
    const categories = app.ruleCategories(RULES, classified);
    if (values['dry-run']) { console.log(JSON.stringify({ reasons: classified.length, bucket: bucket.length, explainer: identity.explainer })); return 0; }
    refuseExisting(values.out ?? `${values.ledger}.rejection-proposals.json`);
    const rt = await runtime();
    try {
      const result = await app.triage(rt, bucket, categories, classified);
      finish('triageRejections', values.out ?? `${values.ledger}.rejection-proposals.json`, { bucket, categories }, result);
    } finally { rt.close?.(); }
    return 0;
  }
  if (command === 'gate') {
    const { facts } = JSON.parse(readFileSync(values.facts, 'utf8'));
    const report = readFileSync(values.report, 'utf8');
    if (values['dry-run']) { console.log(JSON.stringify({ failed: facts.failed.length, explainer: identity.explainer })); return 0; }
    refuseExisting(values.out ?? `${values.report}.explanation.json`);
    const rt = await runtime();
    try {
      const result = await app.explainGate(rt, facts, report.length > 20000 ? `${report.slice(0, 20000)}\n... (${report.length} chars)` : report);
      finish('explainGateFailure', values.out ?? `${values.report}.explanation.json`, { facts, report }, result);
    } finally { rt.close?.(); }
    return 0;
  }
  console.error('usage: explain-advisory.mjs failures|rejections|gate ... (see the file header)');
  return 2;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main(process.argv.slice(2)));
