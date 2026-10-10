#!/usr/bin/env node
/** Method arms over decision families: artifact × signal regime, with compute accounting (LEARNING_CONTINUUM.md
 * §13 M0.2; generalises soft-skill-decision.mjs).
 *
 * Every arm trains (or leaves untrained) one artifact for one family's support cases and is scored on the family's
 * held-out cases (query) and on another family's held-out cases (transfer) with the decision readout's proper scoring
 * rule against gold. Arms:
 *
 *   none             no artifact: the base model's readout;
 *   soft-init        the text-initialised soft skill, untrained;
 *   soft-gold        soft skill, supervised on gold distributions;
 *   soft-teacher     soft skill, distilled from a stronger model's readouts (--teacher-labels; cross-model
 *                    conditioned distillation on decisions, §4.3);
 *   adapter-gold     a weight adapter (`xs`, top-r subspace, layers after the sketch cutoff), supervised on gold;
 *   adapter-teacher  the adapter distilled from the teacher's readouts;
 *   joint-gold       soft skill and adapter trained together on gold;
 *   prompt-gold      the soft form of the runtime's decision system prompt (DECISIONS.md 40), initialised from its
 *                    text and supervised on gold: self-improvement adapting the system prompt for one family;
 *   prompt-teacher   the same, distilled from the teacher's readouts.
 *
 * Each trained arm writes a `natlang.improvement-step/1` record (improvement-steps.jsonl) with its compute: optimiser
 * steps, readout calls during training, wall seconds. The teacher's own quality on the query cases is reported beside
 * the arms. Support cases come from the `train` role, query and transfer from `heldout`.
 *
 * Usage: run-method-arms.mjs --cases decision-cases.jsonl --out DIR --endpoint URL [--families a,b]
 *          [--arms none,soft-init,soft-gold,soft-teacher,adapter-gold,adapter-teacher,joint-gold]
 *          [--teacher-labels labels.jsonl] [--support 16 --query 24 --steps 8 --lr 0.02 --adapter-lr 0.01
 *           --adapter-rank 4 --init-text FILE --init in-context|encode|embed]
 */
import { readFileSync } from 'node:fs';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { iterateOn } from '../../dist/index.js';
import { DECISION_SYSTEM_PROMPT } from '../../dist/native/decision.js';
import { improvementStep } from '../../dist/improvement/step-record.js';
import { caseTarget, casesByFamily, decisionSession, quality, sampleCases } from './decision-lib.mjs';

const ARMS = ['none', 'soft-init', 'soft-gold', 'soft-teacher', 'adapter-gold', 'adapter-teacher', 'joint-gold', 'prompt-gold', 'prompt-teacher'];
const NUMERIC = ['support', 'query', 'steps', 'lr', 'adapter-lr', 'adapter-rank'];
const options = { support: 16, query: 24, steps: 8, lr: 0.02, 'adapter-lr': 0.01, 'adapter-rank': 4, families: '', arms: ARMS.join(','), init: 'in-context', sample: 'stratified' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (![...NUMERIC, 'cases', 'out', 'endpoint', 'families', 'arms', 'teacher-labels', 'init-text', 'init', 'sample'].includes(key) || value === undefined)
    throw Error('Usage: see the header of run-method-arms.mjs');
  options[key] = NUMERIC.includes(key) ? Number(value) : value;
}
if (!options.cases || !options.out || !options.endpoint) throw Error('cases, out and endpoint are required');
const arms = options.arms.split(',');
const unknown = arms.filter(arm => !ARMS.includes(arm));
if (unknown.length) throw Error(`unknown arms ${unknown.join(', ')}; known: ${ARMS.join(', ')}`);
if (arms.some(arm => arm.endsWith('-teacher')) && !options['teacher-labels']) throw Error('teacher arms need --teacher-labels');
const sha = value => createHash('sha256').update(value).digest('hex');
const initText = options['init-text'] ? readFileSync(options['init-text'], 'utf8') :
  'Read the input closely, weigh the evidence for each allowed answer, and give the answer the evidence supports.';

/** The teacher's distribution over a case's values, or undefined when it has no usable label. */
const teacher = new Map();
if (options['teacher-labels'])
  for (const line of readFileSync(options['teacher-labels'], 'utf8').split('\n')) if (line.trim()) {
    const label = JSON.parse(line);
    if (!label.answer?.error) teacher.set(label.id, label.answer);
  }
function teacherTarget(c) {
  const answer = teacher.get(c.id);
  if (!answer) return undefined;
  const { values } = caseTarget(c);
  let probabilities;
  if (c.kind === 'noul' && typeof answer.noul === 'number') probabilities = [answer.noul, 1 - answer.noul];
  else if (c.kind === 'choice' && answer.probabilities) probabilities = values.map(v => Number(answer.probabilities[v] ?? 0));
  else if (c.kind === 'score' && answer.probabilities) probabilities = values.map((_, i) => Number(answer.probabilities[String(i)] ?? 0));
  const total = probabilities?.reduce((sum, p) => sum + p, 0);
  return total > 0 ? probabilities.map(p => p / total) : undefined;
}

const byFamily = casesByFamily(options.cases);
const wanted = options.families ? options.families.split(',') : [...byFamily.keys()].sort();
const teacherArms = arms.some(arm => arm.endsWith('-teacher'));
// With teacher arms, support cases are those the teacher labelled, so gold and teacher arms train on the same cases.
const supportOf = f => sampleCases(byFamily.get(f).train.filter(c => !teacherArms || teacherTarget(c)), options.support, options.sample);
const families = wanted.filter(f => byFamily.get(f) && supportOf(f).length >= options.support && byFamily.get(f).heldout.length >= options.query);
if (families.length < 2) throw Error('need at least two families with enough train and heldout cases');
const split = Object.fromEntries(families.map(f => [f, { support: supportOf(f), query: sampleCases(byFamily.get(f).heldout, options.query, options.sample) }]));

const out = resolve(options.out);
await mkdir(join(out, 'artifacts'), { recursive: true });
await writeFile(join(out, 'run.json'), JSON.stringify({ version: 'natlang.method-arms/1', options, arms, families,
  cases_sha256: sha(readFileSync(options.cases)), teacher_labels_sha256: options['teacher-labels'] ? sha(readFileSync(options['teacher-labels'])) : null,
  init_text_sha256: sha(initText) }, null, 2) + '\n', { flag: 'wx' });

const session = decisionSession(options.endpoint);
const { learning, runtime, readouts, readoutOf, lossOn: lossWith } = session;
// Soft artifacts start from their text in context (text-init.ts, owner 2026-10-10: the soft call reproduces the
// text-instructed call), or as diagnostic context-free arms `--init encode` (through the port, method-arms v1/v3) and
// `--init embed` (raw token embeddings, v2).
if (!['in-context', 'encode', 'embed'].includes(options.init)) throw Error('--init is in-context, encode or embed');
const initialise = async (text, c) => options.init === 'in-context' ? (await session.initSkill(text, c)).skill
  : options.init === 'embed' ? session.embed(text) : session.encode(text);
const { valueAndGrad, optimizers, adapters } = learning;
const lossOn = (cases, source) => lossWith(cases, source === 'teacher' ? teacherTarget : c => caseTarget(c).gold);

/** Adam on `loss(params)`; adapter leaves get their own learning rate (a separate optimiser over the same steps). */
async function tune(start, loss) {
  const soft = optimizers.adam({ lr: options.lr }), weights = optimizers.adam({ lr: options['adapter-lr'] });
  const trace = [];
  const readoutsBefore = readouts(), started = Date.now();
  const step = async state => {
    const { loss: value, grad } = await valueAndGrad(loss, state.value);
    trace.push(+value);
    const next = { ...state.value }, opt = { ...state.opt };
    for (const key of Object.keys(state.value)) {
      const optimizer = key === 'adapter' ? weights : soft;
      const moved = await optimizer.step({ value: state.value[key], opt: state.opt[key] }, { $gradient: grad.$gradient[key] });
      next[key] = moved.value; opt[key] = moved.opt;
    }
    return { value: next, opt };
  };
  const init = { value: start, opt: Object.fromEntries(Object.keys(start).map(key => [key, (key === 'adapter' ? weights : soft).init(start[key])])) };
  const final = await runtime.run(() => iterateOn(step, init).withLimit({ maxSteps: options.steps }).checkProgress('off').until(() => false))
    .catch(error => { if (error?.name === 'IterationLimitError') return error.lastState; throw error; });
  return { value: final.value, trace, compute: { gradient_steps: trace.length, readout_calls: readouts() - readoutsBefore,
    seconds: (Date.now() - started) / 1000 } };
}
async function evaluate(cases, params) {
  let total = 0;
  for (const c of cases) total += quality(c, await readoutOf(c, params), caseTarget(c).gold);
  return total / cases.length;
}
const teacherQuality = cases => {
  const scored = cases.map(c => [c, teacherTarget(c)]).filter(([, p]) => p);
  return scored.length ? { quality: scored.reduce((sum, [c, p]) => sum + quality(c, p, caseTarget(c).gold), 0) / scored.length, cases: scored.length } : null;
};

const record = entry => appendFile(join(out, 'results.jsonl'), JSON.stringify(entry) + '\n');
const stepRecord = entry => appendFile(join(out, 'improvement-steps.jsonl'), JSON.stringify(improvementStep(entry)) + '\n');
const init = await initialise(initText, supportOf(families[0])[0]);
const refs = params => [...(params.skill ? [{ kind: 'soft-skill', id: params.skill.$neuralese.id, role: 'skill' }] : []),
  ...(params.adapter ? [{ kind: 'adapter', id: params.adapter.$neuralese.id, role: 'adapter' }] : []),
  ...Object.entries(params.prompts ?? {}).map(([piece, ref]) => ({ kind: 'system-prompt', id: ref.$neuralese.id, role: `prompt:${piece}` }))];

for (const [index, family] of families.entries()) {
  const other = families[(index + 1) % families.length];
  const { support, query } = split[family], transfer = split[other].query;
  const scores = {}, steps = {}, artifacts = {};
  for (const arm of arms) {
    const [artifact, source] = arm.split('-');
    let start = {}, params = {}, tuned = null;
    if (artifact === 'soft') start = { skill: init };
    if (artifact === 'adapter') start = { adapter: await adapters.create({ kind: 'xs', rank: options['adapter-rank'] }) };
    if (artifact === 'joint') start = { skill: init, adapter: await adapters.create({ kind: 'xs', rank: options['adapter-rank'] }) };
    if (artifact === 'prompt') start = { prompts: { decision: await initialise(DECISION_SYSTEM_PROMPT) } };
    if (source === 'gold' || source === 'teacher') {
      tuned = await tune(start, lossOn(support, source));
      params = tuned.value;
    } else params = start;
    scores[arm] = { query: await evaluate(query, params), transfer: await evaluate(transfer, params) };
    if (tuned) {
      steps[arm] = tuned.trace;
      if (params.skill) artifacts[`${arm.replace('-', '_')}_skill`] = { type: 'Neuralese<string>', value: params.skill };
      if (params.adapter) artifacts[`${arm.replace('-', '_')}_adapter`] = { type: 'Adapter', value: params.adapter };
      if (params.prompts) artifacts[`${arm.replace('-', '_')}_decision_prompt`] = { type: 'Neuralese<string>', value: params.prompts.decision };
      await stepRecord({ episode: { id: `method-arms:${family}`, family }, facets: [`family:${family}`, `artifact:${artifact}`, `regime:${source}`],
        before: refs(start), operator: { kind: `method-arm:${arm}`, version: 'run-method-arms/1',
          regime: source === 'teacher' ? 'conditioned-distillation' : 'supervised',
          hyper: { optimizer: 'adam', lr: options.lr, adapterLr: options['adapter-lr'], adapterRank: options['adapter-rank'], steps: options.steps,
            support: support.length, ...(source === 'teacher' ? { teacher: [...teacher.values()][0]?.teacher ?? options['teacher-labels'] } : {}) },
          context: null, model: null },
        view: { visibility: 'full', evidence: [] },
        proposal: { deltas: refs(params).map(artifactRef => ({ artifact: artifactRef, delta: null, scale: 1 })) },
        after: refs(params),
        outcome: { query: { before: null, after: scores[arm].query, effect: null }, transfer: { before: null, after: scores[arm].transfer, effect: null },
          compute: { gradient_steps: tuned.compute.gradient_steps, model_calls: tuned.compute.readout_calls, seconds: tuned.compute.seconds } },
        trajectory: { id: `method-arms:${family}:${arm}`, step: 0 } });
    }
    console.log(JSON.stringify({ family, arm, ...scores[arm], ...(tuned ? { loss: [tuned.trace[0], tuned.trace.at(-1)], compute: tuned.compute } : {}) }));
  }
  await record({ family, transfer_family: other, scores, traces: steps, teacher: { query: teacherQuality(query), transfer: teacherQuality(transfer) } });
  // The trained artifacts, so the improvement steps stay resolvable after the server's in-memory store is gone.
  if (Object.keys(artifacts).length) await learning.save(join(out, 'artifacts', `${family.replace(/[^a-z0-9-]+/gi, '_')}.nz`), artifacts);
}
