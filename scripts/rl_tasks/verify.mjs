#!/usr/bin/env node
// Qualify broad external RL cases through the collector's own execution path (plans/neuralese/BROAD_RL_TASKS_20261031.md).
// usage: node scripts/rl_tasks/verify.mjs RUNTIME_TS_HOST IN.jsonl OUT_DIR [CONCURRENCY]
// Per case: (1) verifyCases: validation plus the reference replayed to an admitted, accepted result; (2) negative
// controls: the reference's terminal answer replaced by each declared meaningful wrong answer (other items' gold of the
// same source) and by a generic perturbation must be rejected; (3) false-negative probe: each declared alternative gold
// form (upstream aliases/annotator answers, trivial surface variants) replayed and its acceptance recorded. A case is kept
// when (1) passes and every negative control is rejected. Alias acceptance is measured, never a keep condition.
// Writes OUT_DIR/cases.jsonl (kept rows, input order) and OUT_DIR/verification.json.
import fs from 'node:fs';
import crypto from 'node:crypto';
const [runtime, input, out, concurrencyArg] = process.argv.slice(2);
const { verifyCases, replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const { externalModule } = await import(runtime + '/dist/native/external.js');
const { createRequire } = await import('node:module');
const requireModule = createRequire(runtime + '/dist/native/external.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
// Service-certified answers (tool calls): the expected value, wrong answers and alternative gold forms are the
// receipts the case's own service returns for the gold, wrong and alias call sets.
async function materializeCallSets(record) {
  const sets = record.curriculum?.external?.call_sets;
  if (!sets) return;
  const api = externalModule(sets.service, record.semantics.services[sets.service], requireModule).exports;
  const receipts = async calls => { const out = []; for (const call of calls) out.push(await api[sets.names[call.name]](call.arguments ?? {})); return out; };
  record.semantics.expected = await receipts(sets.gold);
  const checks = record.curriculum.external.checks;
  const goldKey = JSON.stringify([...record.semantics.expected].sort());
  checks.wrong_answers = [];
  for (const calls of sets.wrong) { const value = await receipts(calls); if (JSON.stringify([...value].sort()) !== goldKey) checks.wrong_answers.push(value); }
  checks.alias_answers = [];
  for (const calls of sets.alias) checks.alias_answers.push(await receipts(calls));
}
for (const record of rows) await materializeCallSets(record);
const concurrency = Number(concurrencyArg || 8);
const perturb = v => {
  if (typeof v === 'string') return v + ' and something else entirely';
  if (typeof v === 'number') return v + 7;
  if (typeof v === 'boolean') return !v;
  if (v === null) return 'none';
  if (Array.isArray(v)) return v.length > 1 ? v.slice(0, -1) : v.length ? [...v, v[0]] : ['unexpected'];
  if (typeof v === 'object') {
    const keys = Object.keys(v);
    if (!keys.length) return { unexpected: true };
    return { ...v, [keys[0]]: perturb(v[keys[0]]) };
  }
  return 'unexpected';
};
// The terminal answering action of a reference: a return_result value, or the literal of a final `return <JSON>;` eval.
function withAnswer(record, value) {
  const copy = structuredClone(record);
  const root = copy.curriculum.reference.root;
  const terminal = root.findLast(x => x[0] === 'return_result' && x[1]?.status === 'success');
  if (terminal) { terminal[1].value = value; return copy; }
  const last = root[root.length - 1];
  if (last?.[0] === 'eval') { last[1] = { ...last[1], code: 'return ' + JSON.stringify(value) + ';' }; return copy; }
  return undefined;
}
async function accepted(record) {
  const { run } = await replayReference(record, TOOLS_PROMPT);
  return run.outcome.accepted === true;
}
function terminalValue(record) {
  const root = record.curriculum.reference.root;
  const terminal = root.findLast(x => x[0] === 'return_result' && x[1]?.status === 'success');
  if (terminal) return { found: true, value: terminal[1].value };
  const m = /^\s*return\s+([\s\S]*?);?\s*$/.exec(root[root.length - 1]?.[1]?.code || '');
  if (m) { try { return { found: true, value: JSON.parse(m[1]) }; } catch {} }
  return { found: false };
}
async function qualify(record) {
  const result = { id: record.id, family: record.family, split: record.split, shape: record.curriculum?.shape,
    reference_ok: false, problems: [], wrong: [], aliases: [] };
  try {
    const [ref] = await verifyCases([record], TOOLS_PROMPT);
    result.reference_ok = ref.ok; result.problems = ref.problems;
  } catch (e) { result.problems.push(String(e).slice(0, 400)); }
  const checks = record.curriculum?.external?.checks ?? {};
  const terminal = terminalValue(record);
  const wrongs = [...(checks.wrong_answers ?? [])];
  if (terminal.found && !checks.wrong_code) wrongs.push(perturb(terminal.value));
  // Effect-graded cases (authoring): negative controls replace the final eval's code with a wrong implementation.
  for (const code of checks.wrong_code ?? []) {
    const variant = structuredClone(record);
    const last = variant.curriculum.reference.root.at(-1);
    last[1] = { ...last[1], code };
    try { result.wrong.push({ value: code.slice(0, 120), accepted: await accepted(variant) }); }
    catch (e) { result.wrong.push({ value: code.slice(0, 120), accepted: null, error: String(e).slice(0, 300) }); }
  }
  for (const value of wrongs) {
    const variant = withAnswer(record, value);
    if (!variant) { result.wrong.push({ value, accepted: null, error: 'no terminal answer to replace' }); continue; }
    try { result.wrong.push({ value, accepted: await accepted(variant) }); }
    catch (e) { result.wrong.push({ value, accepted: null, error: String(e).slice(0, 300) }); }
  }
  for (const value of checks.alias_answers ?? []) {
    const variant = withAnswer(record, value);
    if (!variant) continue;
    try { result.aliases.push({ value, accepted: await accepted(variant) }); }
    catch (e) { result.aliases.push({ value, accepted: null, error: String(e).slice(0, 300) }); }
  }
  result.wrong_rejected = result.wrong.length > 0 && result.wrong.every(w => w.accepted === false);
  result.ok = result.reference_ok && result.wrong_rejected;
  return result;
}
const results = new Array(rows.length);
let next = 0, done = 0;
async function worker() {
  while (next < rows.length) {
    const index = next++;
    results[index] = await qualify(rows[index]);
    if (++done % 50 === 0) console.error(JSON.stringify({ done, of: rows.length }));
  }
}
await Promise.all(Array.from({ length: concurrency }, worker));
const kept = rows.filter((_, i) => results[i].ok);
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(out + '/cases.jsonl', kept.map(r => JSON.stringify(r) + '\n').join(''));
fs.writeFileSync(out + '/materialized-input.jsonl', rows.map(r => JSON.stringify(r) + '\n').join(''));
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const rate = (n, d) => d ? Number((n / d).toFixed(4)) : null;
const aliasTried = results.flatMap(r => r.aliases).filter(a => a.accepted !== null);
const summary = {
  cases: rows.length, kept: kept.length,
  reference_ok: results.filter(r => r.reference_ok).length,
  wrong_controls: results.reduce((n, r) => n + r.wrong.length, 0),
  wrong_accepted: results.reduce((n, r) => n + r.wrong.filter(w => w.accepted === true).length, 0),
  wrong_errors: results.reduce((n, r) => n + r.wrong.filter(w => w.accepted === null).length, 0),
  alias_probes: aliasTried.length, alias_rejected: aliasTried.filter(a => !a.accepted).length,
  alias_false_negative_rate: rate(aliasTried.filter(a => !a.accepted).length, aliasTried.length),
  by_split: Object.fromEntries(['train', 'dev', 'confirmation'].map(s => [s, kept.filter(r => r.split === s).length])),
};
const report = { schema: 'natlang.broad-rl-case-verification/1', input: { path: input, sha256: sha(input) }, runtime,
  cases_sha256: sha(out + '/cases.jsonl'), summary,
  failures: results.filter(r => !r.ok).slice(0, 200).map(r => ({ id: r.id, problems: r.problems.slice(0, 3),
    wrong_accepted: r.wrong.filter(w => w.accepted !== false).map(w => w.value) })),
  alias_rejections: results.flatMap(r => r.aliases.filter(a => a.accepted === false).map(a => ({ id: r.id, value: a.value }))).slice(0, 200),
  results };
fs.writeFileSync(out + '/verification.json', JSON.stringify(report) + '\n');
console.log(JSON.stringify(summary));
