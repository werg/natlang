#!/usr/bin/env node
/** Soft skills on decision families, scored by proper scoring rules on the decision readout (S6 baseline, v2).
 *
 * The SQL soft-skill baseline showed two problems: gains were mostly generic answer-format priming (transfer cases
 * of another family improved almost as much), and a 350M executor never wrote correct SQL, so likelihood gains never
 * reached graded quality. Here every case is a typed decision (`readout: decision`), so each arm's quality is a
 * calibrated distribution scored directly (Brier for choices and yes/no, ranked probability score for ordered
 * levels), and the arms separate family-specific learning from generic priming:
 *
 *   none       no skill item;
 *   text-init  the skill block initialised from a crisp skill text's token embeddings;
 *   generic    text-init tuned on support cases pooled from every family in the run: the format-priming reference;
 *   tuned      text-init tuned on the family's own support cases;
 *   specific   generic tuned on the family's own support cases plus a term that holds its readout on other
 *              families' cases to the generic skill's (cross-entropy against the fixed generic distribution, i.e.
 *              KL(generic ‖ skill) up to a constant), so it cannot win by more priming.
 *
 * Gains are reported against none and against generic, on the family's held-out cases (query) and on another
 * family's held-out cases (transfer). Specificity = query gain over generic − transfer gain over generic. Support
 * and contrast cases come from the `train` role; query and transfer from `heldout`; no held-out case enters a
 * gradient.
 *
 * Usage: soft-skill-decision.mjs --cases decision-cases.jsonl --out DIR --endpoint URL [--families a,b,...]
 *          [--support 16 --query 24 --contrast 8 --steps 8 --lr 0.02 --kl-weight 1 --init-text FILE]
 */
import { readFileSync } from 'node:fs';
import { improvementStep } from '../../dist/improvement/step-record.js';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { Context, createNatlangRuntime, iterateOn, learningService, createLearning, loadVirtualNatlang } from '../../dist/index.js';
import { MemoryNeuraleseStore } from '../../dist/native/neuralese-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { rankedProbabilityScore } from '../../dist/skills/graded.js';
import { sampleCases } from './decision-lib.mjs';

const options = { support: 16, query: 24, contrast: 8, steps: 8, lr: 0.02, 'kl-weight': 1, families: '', sample: 'stratified' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (!['cases', 'out', 'endpoint', 'families', 'support', 'query', 'contrast', 'steps', 'lr', 'kl-weight', 'init-text', 'sample'].includes(key) || value === undefined)
    throw Error('Usage: see the header of soft-skill-decision.mjs');
  options[key] = ['support', 'query', 'contrast', 'steps', 'lr', 'kl-weight'].includes(key) ? Number(value) : value;
}
if (!options.cases || !options.out || !options.endpoint) throw Error('cases, out and endpoint are required');
const sha = value => createHash('sha256').update(value).digest('hex');
const initText = options['init-text'] ? readFileSync(options['init-text'], 'utf8') :
  'Read the input closely, weigh the evidence for each allowed answer, and give the answer the evidence supports.';

/** Values of a case's result type and the gold distribution over them (as export-decision-prompts.mjs). */
function caseTarget(c) {
  if (c.kind === 'choice') return { values: c.options, gold: c.options.map(v => typeof c.answer === 'string' ? Number(v === c.answer) : Number(c.answer?.[v] ?? 0)) };
  if (c.kind === 'noul') { const p = Number(c.answer); return { values: [true, false], gold: [p, 1 - p] }; }
  const x = typeof c.answer === 'string' && c.levels.includes(c.answer) ? c.levels.indexOf(c.answer) : Number(c.answer);
  const gold = new Array(c.levels.length).fill(0), low = Math.floor(x), frac = x - low;
  gold[low] += 1 - frac; if (frac > 0) gold[low + 1] += frac;
  return { values: c.levels, gold };
}
const quality = (c, predicted, gold) => c.kind === 'score' ? 1 - rankedProbabilityScore(predicted, gold)
  : 1 - predicted.reduce((sum, p, i) => sum + (p - gold[i]) ** 2, 0) / 2;

// Cases by family and role; the first `n` of each in file order (the builder already shuffled within a family).
const byFamily = new Map();
for (const line of readFileSync(options.cases, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const c = JSON.parse(line);
  if (c.kind === 'choice' && c.options.length > 12) continue;  // keep each readout small for the 350M server
  const entry = byFamily.get(c.family) ?? { train: [], heldout: [] };
  entry[c.role]?.push(c);
  byFamily.set(c.family, entry);
}
const wanted = options.families ? options.families.split(',') : [...byFamily.keys()].sort();
const families = wanted.filter(f => byFamily.get(f)?.train.length >= options.support && byFamily.get(f)?.heldout.length >= options.query);
if (families.length < 2) throw Error('need at least two families with enough train and heldout cases');
const split = Object.fromEntries(families.map(f => {
  const support = sampleCases(byFamily.get(f).train, options.support, options.sample);
  return [f, { support, contrast: sampleCases(byFamily.get(f).train.filter(c => !support.includes(c)), options.contrast, options.sample),
    query: sampleCases(byFamily.get(f).heldout, options.query, options.sample) }];
}));

const out = resolve(options.out);
await mkdir(out, { recursive: true });
await writeFile(join(out, 'run.json'), JSON.stringify({ version: 'natlang.soft-skill-decision/1', options, families,
  cases_sha256: sha(readFileSync(options.cases)), init_text_sha256: sha(initText) }, null, 2) + '\n', { flag: 'wx' });

const store = new MemoryNeuraleseStore();
const service = learningService({ endpoint: options.endpoint, store });
const { valueAndGrad, objectives, optimizers } = createLearning(service);
const traces = [];
const runtime = createNatlangRuntime({ model: { driver: neuraleseServerModelTurn({ endpoint: options.endpoint, model: 'natlang-neuralese', store }) },
  neuralese: { store }, seed: { mode: 'backend' }, trace: trace => traces.push(trace) });
const functions = new Map();
/** The case as a typed decision function, bound to a context holding the skill item when there is one. */
function call(c, skill) {
  let fn = functions.get(c.id);
  if (!fn) {
    const { values } = caseTarget(c);
    const returns = c.kind === 'noul' ? 'boolean' : values.map(v => JSON.stringify(v)).join(' | ');
    fn = loadVirtualNatlang({ 'decide.nl': `---\nargs: { state: string }\nreturns: ${returns}\nreadout: decision\n---\n${c.question}\n` }, 'decide.nl');
    functions.set(c.id, fn);
  }
  const bound = skill ? fn.in(Context.ofCallable(fn).with({ skill })) : fn;
  return runtime.run(() => bound(c.state));
}
const target = c => { const { values, gold } = caseTarget(c); return Object.fromEntries(values.map((v, i) => [String(v), gold[i]])); };
/** Map with at most `limit` calls in flight: the reference server queues requests on one engine, and a hundred
 * simultaneous connections exhaust its listener. */
async function bounded(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
}
const decisionLoss = (cases, skill) => bounded(cases, 4, c => objectives.decision(() => call(c, skill), target(c), 'logLoss'))
  .then(losses => objectives.sum(...losses));

async function embed(text) {
  const response = await fetch(`${options.endpoint}/v1/neuralese/embed`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, type: 'Neuralese<string>' }) });
  if (!response.ok) throw Error(`embed failed: ${response.status} ${await response.text()}`);
  return { $neuralese: { type: 'Neuralese<string>', id: (await response.json()).id } };
}

/** Adam on `loss(skill)` from `start`; returns the tuned value and the loss trace. */
async function tune(start, loss) {
  const adam = optimizers.adam({ lr: options.lr });
  const trace = [];
  const step = async state => {
    const { loss: value, grad } = await valueAndGrad(loss, state.value);
    trace.push(+value);
    return adam.step(state, grad);
  };
  const final = await runtime.run(() => iterateOn(step, { value: start, opt: adam.init(start) }).withLimit({ maxSteps: options.steps })
    .checkProgress('off').until(() => false)).catch(error => {
    if (error?.name === 'IterationLimitError') return error.lastState;
    throw error;
  });
  return { value: final.value, trace };
}

/** The readout distribution of `skill` on one case. */
async function readoutOf(c, skill) {
  const before = traces.length;
  await call(c, skill);
  const event = traces.slice(before).flatMap(trace => trace.events).find(item => Array.isArray(item.probabilities) && item.options);
  if (!event) throw Error('no decision readout recorded for ' + c.id);
  return event.probabilities;
}

/** Mean readout quality and log loss of `skill` on cases (no gradient). */
async function evaluate(cases, skill) {
  let total = 0, logLoss = 0;
  for (const c of cases) {
    const probabilities = await readoutOf(c, skill);
    const { gold } = caseTarget(c);
    total += quality(c, probabilities, gold);
    logLoss -= gold.reduce((sum, g, i) => sum + (g ? g * Math.log(Math.max(probabilities[i], 1e-12)) : 0), 0);
  }
  return { quality: total / cases.length, log_loss: logLoss / cases.length };
}

const record = entry => appendFile(join(out, 'results.jsonl'), JSON.stringify(entry) + '\n');
/** One `natlang.improvement-step/1` record per trained arm (LEARNING_CONTINUUM.md §8), with the blocks' IDs. */
const step = (family, arm, from, to, trace, gains) => appendFile(join(out, 'improvement-steps.jsonl'), JSON.stringify(improvementStep({
  episode: { id: `soft-skill-decision:${family}`, family }, facets: [`family:${family}`, `operator:soft-skill-${arm}`, 'artifact:soft-skill'],
  before: [{ kind: 'soft-skill', id: from.$neuralese.id, role: 'skill' }],
  operator: { kind: `soft-skill-${arm}`, version: 'soft-skill-decision/4', regime: arm === 'specific' ? 'conditioned-distillation' : 'supervised',
    hyper: { optimizer: 'adam', lr: options.lr, steps: options.steps, support: options.support,
      ...(arm === 'specific' ? { klWeight: options['kl-weight'], contrast: options.contrast } : {}) }, context: null, model: null },
  view: { visibility: 'full', evidence: [] },
  proposal: { deltas: [{ artifact: { kind: 'soft-skill', id: to.$neuralese.id, role: 'skill' }, delta: null, scale: 1 }] },
  after: [{ kind: 'soft-skill', id: to.$neuralese.id, role: 'skill' }],
  outcome: { ...gains, compute: { gradient_steps: trace.length } },
  trajectory: { id: `soft-skill-decision:${family}:${arm}`, step: 0 } })) + '\n');
const init = await embed(initText);
// The generic skill: support pooled across every family, an equal share from each.
const share = Math.max(1, Math.floor(options.support / 2));
const pooled = families.flatMap(f => split[f].support.slice(0, share));
const generic = await tune(init, skill => decisionLoss(pooled, skill));
await record({ arm: 'generic', pooled: pooled.length, trace: generic.trace, block: generic.value.$neuralese.id });

for (const [index, family] of families.entries()) {
  const other = families[(index + 1) % families.length];
  const { support, query } = split[family];
  const transfer = split[other].query;
  // Contrast cases come from every other family's train role, never from held-out cases.
  const contrast = families.filter(f => f !== family).flatMap(f => split[f].contrast).slice(0, options.contrast);
  const own = await tune(init, skill => decisionLoss(support, skill));
  // The generic skill's readout on the contrast cases, fixed: cross-entropy against it is KL(generic ‖ skill) plus a
  // constant, so the term is bounded below and only holds the skill to generic behaviour off its family.
  const anchors = [];
  for (const c of contrast) {
    const { values } = caseTarget(c), readout = await readoutOf(c, generic.value);
    anchors.push(Object.fromEntries(values.map((v, i) => [String(v), readout[i]])));
  }
  const specific = await tune(generic.value, async skill => {
    const fit = await decisionLoss(support, skill);
    if (!contrast.length || !options['kl-weight']) return fit;
    const keep = await bounded(contrast, 4, (c, i) => objectives.decision(() => call(c, skill), anchors[i], 'logLoss'));
    return objectives.sum(fit, await objectives.scale(await objectives.sum(...keep), options['kl-weight'] / contrast.length));
  });
  const arms = { none: undefined, 'text-init': init, generic: generic.value, tuned: own.value, specific: specific.value };
  const scores = {};
  for (const [name, skill] of Object.entries(arms))
    scores[name] = { query: await evaluate(query, skill), transfer: await evaluate(transfer, skill) };
  const gain = (arm, ref, where) => scores[arm][where].quality - scores[ref][where].quality;
  const summary = { family, transfer_family: other, scores, traces: { tuned: own.trace, specific: specific.trace },
    gains: Object.fromEntries(['text-init', 'generic', 'tuned', 'specific'].map(arm => [arm, {
      query_vs_none: gain(arm, 'none', 'query'), transfer_vs_none: gain(arm, 'none', 'transfer'),
      query_vs_generic: gain(arm, 'generic', 'query'), transfer_vs_generic: gain(arm, 'generic', 'transfer'),
      specificity: gain(arm, 'generic', 'query') - gain(arm, 'generic', 'transfer') }])) };
  await record(summary);
  // Gains are measured from each step's own starting point: text-init for tuned, generic for specific.
  const gainOf = (arm, from) => Object.fromEntries(['query', 'transfer'].map(where => [where, { before: scores[from][where].quality,
    after: scores[arm][where].quality, effect: scores[arm][where].quality - scores[from][where].quality }]));
  if (index === 0) await step('pooled', 'generic', init, generic.value, generic.trace, {});
  await step(family, 'tuned', init, own.value, own.trace, gainOf('tuned', 'text-init'));
  await step(family, 'specific', generic.value, specific.value, specific.trace, gainOf('specific', 'generic'));
  console.log(JSON.stringify({ family, gains: summary.gains }));
}
