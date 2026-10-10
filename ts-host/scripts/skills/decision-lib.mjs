/** Shared pieces of the decision-family learning scripts (run-method-arms.mjs, memetic-decision.mjs): case targets,
 * scoring, case splits, the runtime against a Neuralese server, and typed decision calls with optional guidance text,
 * soft skill and adapter. */
import { readFileSync } from 'node:fs';
import { Context, createNatlangRuntime, learningService, createLearning, loadVirtualNatlang, initBodyInContext } from '../../dist/index.js';
import { MemoryNeuraleseStore } from '../../dist/native/neuralese-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { rankedProbabilityScore } from '../../dist/skills/graded.js';

/** Values of a case's result type and the gold distribution over them (as export-decision-prompts.mjs). */
export function caseTarget(c) {
  if (c.kind === 'choice') return { values: c.options, gold: c.options.map(v => typeof c.answer === 'string' ? Number(v === c.answer) : Number(c.answer?.[v] ?? 0)) };
  if (c.kind === 'noul') { const p = Number(c.answer); return { values: [true, false], gold: [p, 1 - p] }; }
  const x = typeof c.answer === 'string' && c.levels.includes(c.answer) ? c.levels.indexOf(c.answer) : Number(c.answer);
  const gold = new Array(c.levels.length).fill(0), low = Math.floor(x), frac = x - low;
  gold[low] += 1 - frac; if (frac > 0) gold[low + 1] += frac;
  return { values: c.levels, gold };
}

/** Quality of a predicted distribution: 1 − Brier/2 for choices and yes/no, 1 − RPS for ordered levels. */
export const quality = (c, predicted, gold) => c.kind === 'score' ? 1 - rankedProbabilityScore(predicted, gold)
  : 1 - predicted.reduce((sum, p, i) => sum + (p - gold[i]) ** 2, 0) / 2;

/** The answer a case is stratified by: a choice's answer, a yes/no's rounded probability, a score's level. */
const stratum = c => c.kind === 'choice' ? (typeof c.answer === 'string' ? c.answer : JSON.stringify(c.answer))
  : c.kind === 'noul' ? String(Number(c.answer) >= 0.5) : String(c.answer);

/**
 * `n` cases taken `first` in file order, or `stratified`: round-robin over the answers (in file order within each),
 * so a small support or query set holds the answers in proportions as equal as the cases allow. With a handful of
 * cases per set, file order can put most of one answer in support and another in query (trec-question: 5/16
 * "human beings" in support, 2/24 in query), and tuning then fits the support's answer distribution.
 */
export function sampleCases(cases, n, how = 'stratified') {
  if (how === 'first') return cases.slice(0, n);
  if (how !== 'stratified') throw Error(`--sample is first or stratified, not ${how}`);
  const groups = new Map();
  for (const c of cases) (groups.get(stratum(c)) ?? groups.set(stratum(c), []).get(stratum(c))).push(c);
  const queues = [...groups.values()], out = [];
  for (let round = 0; out.length < n && queues.some(q => q.length > round); round++)
    for (const queue of queues) if (out.length < n && round < queue.length) out.push(queue[round]);
  return out;
}

/** Cases by family and role (`train`, `heldout`) in file order; large choice sets are left out. */
export function casesByFamily(path, maxOptions = 12) {
  const byFamily = new Map();
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const c = JSON.parse(line);
    if (c.kind === 'choice' && c.options.length > maxOptions) continue;
    const entry = byFamily.get(c.family) ?? { train: [], heldout: [] };
    entry[c.role]?.push(c);
    byFamily.set(c.family, entry);
  }
  return byFamily;
}

/** Map with at most `limit` calls in flight. */
export async function bounded(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const i = next++; results[i] = await fn(items[i], i); }
  }));
  return results;
}

/** A runtime and learning surface against a Neuralese server, with typed decision calls. */
export function decisionSession(endpoint) {
  const store = new MemoryNeuraleseStore();
  const learning = createLearning(learningService({ endpoint, store }));
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver: neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store }) },
    neuralese: { store }, seed: { mode: 'backend' }, trace: trace => traces.push(trace) });
  const functions = new Map();
  /** The case as a typed decision function: `params.guidance` (crisp text before the question), `params.skill` (a soft
   * context item), `params.adapter` (active weight adapter) and `params.prompts` (soft system-prompt pieces by piece ID,
   * DECISIONS.md 40) are each optional. */
  function call(c, params = {}, on = runtime) {
    const key = c.id + '\0' + (params.guidance ?? '');
    let fn = functions.get(key);
    if (!fn) {
      const { values } = caseTarget(c);
      const returns = c.kind === 'noul' ? 'boolean' : values.map(v => JSON.stringify(v)).join(' | ');
      const body = (params.guidance ? params.guidance.trim() + '\n\n' : '') + c.question;
      fn = loadVirtualNatlang({ 'decide.nl': `---\nargs: { state: string }\nreturns: ${returns}\nreadout: decision\n---\n${body}\n` }, 'decide.nl');
      functions.set(key, fn);
    }
    const bound = params.skill ? fn.in(Context.ofCallable(fn).with({ skill: params.skill })) : fn;
    const run = () => on.run(() => bound(c.state));
    const prompted = params.prompts ? () => learning.withSystemPrompts(params.prompts, run) : run;
    return params.adapter ? learning.withAdapters(params.adapter, prompted) : prompted();
  }
  const readouts = () => traces.reduce((n, trace) => n + trace.events.filter(item => Array.isArray(item.probabilities) && item.options).length, 0);
  /** The readout distribution of `params` on one case. */
  async function readoutOf(c, params) {
    const before = traces.length;
    await call(c, params);
    const event = traces.slice(before).flatMap(trace => trace.events).find(item => Array.isArray(item.probabilities) && item.options);
    if (!event) throw Error('no decision readout recorded for ' + c.id);
    return event.probabilities;
  }
  async function embed(text, type = 'Neuralese<string>') {
    const response = await fetch(`${endpoint}/v1/neuralese/embed`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, type }) });
    if (!response.ok) throw Error(`embed failed: ${response.status} ${await response.text()}`);
    return { $neuralese: { type, id: (await response.json()).id } };
  }
  /** A block encoding `text` in one forward pass through the port (one vector per token; no summarising call). */
  async function encode(text, type = 'Neuralese<string>') {
    const response = await fetch(`${endpoint}/v1/neuralese/encode`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, type }) });
    if (!response.ok) throw Error(`encode failed: ${response.status} ${await response.text()}`);
    return { $neuralese: { type, id: (await response.json()).id } };
  }
  /**
   * A soft skill initialised in context from `text` (src/neuralese/text-init.ts, owner 2026-10-10): case `c`'s call is
   * rendered with the skill as a placeholder block, and the server returns the body under which that call is the call
   * with the skill as text (gated). The body depends on the context only through tokenization at its edges, so one case
   * serves a family. Returns the skill and the init record (gate, exactness).
   */
  async function initSkill(text, c, type = 'Neuralese<string>') {
    const result = await initBodyInContext({ endpoint, text, type,
      render: (placeholder, driver) => call(c, { skill: { $neuralese: { type, id: placeholder } } },
        createNatlangRuntime({ model: { driver }, neuralese: { store }, seed: { mode: 'backend' } })) });
    return { skill: { $neuralese: { type, id: result.id } }, init: { case: c.id, context: result.context, ...result.init, gate: result.gate } };
  }
  const asTarget = (c, probabilities) => Object.fromEntries(caseTarget(c).values.map((v, i) => [String(v), probabilities[i]]));
  /** Summed decision log loss of `params` on cases against gold (or `targetOf(c)`), as a learning Loss. */
  const lossOn = (cases, targetOf = c => caseTarget(c).gold) => params => bounded(cases, 4, c =>
    learning.objectives.decision(() => call(c, params), asTarget(c, targetOf(c)), 'logLoss')).then(losses => learning.objectives.sum(...losses));
  return { store, learning, runtime, traces, call, readouts, readoutOf, embed, encode, initSkill, lossOn };
}
