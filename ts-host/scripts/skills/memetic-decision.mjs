#!/usr/bin/env node
/** A first memetic optimiser (LEARNING_CONTINUUM.md §7.2) over decision families. An individual is a crisp guidance
 * text and a soft skill block; operators are drawn by a UCB bandit credited with validation gain per second:
 *
 *   propose         reflective rewrite of the guidance by an author model (Qwen), shown the individual's worst
 *                   support cases with its readout and their gradient share (a gradient digest), soft part kept;
 *   embed           bridge crisp → soft: the soft skill re-initialised from the guidance text;
 *   refine-lamarck  k Adam steps on the soft skill; the child keeps the refined block;
 *   refine-baldwin  the same refinement scored, but the child keeps the unrefined block (selects for refinability);
 *   merge           two individuals whose soft skills share an origin: learned merge coefficients over their deltas.
 *
 * Selection keeps the per-case Pareto front on validation cases (as GEPA) up to the population size, then the best by
 * mean. Fitness is measured on validation cases (train role, disjoint from the support cases used by gradients and the
 * author): by default readout quality (as the final score; bounded per case), or with `--fitness logloss` the decision
 * log loss, which a few confidently wrong cases dominate. Seeds are the generic soft-init text of the method arms
 * (no guidance) and author-written guidance texts, each with its embedding. The final best individual is scored on
 * held-out query and transfer cases by readout quality, next to two references on the same split: the seed and
 * `soft-gold` (the method arm: `--reference-steps` Adam steps on the generic soft-init), with compute for each.
 * Each operator application writes a `natlang.improvement-step/1` record.
 *
 * Usage: memetic-decision.mjs --cases decision-cases.jsonl --out DIR --endpoint URL --families a,b [--author-endpoint
 *          URL --author-model ID --generations 24 --population 6 --support 16 --validation 16 --query 24 --steps 8
 *          --lr 0.02 --seed-texts 3 --guidance-words 60 --fitness quality|logloss --reference-steps 8 --init encode|embed]
 */
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { iterateOn } from '../../dist/index.js';
import { improvementStep } from '../../dist/improvement/step-record.js';
import { blockFloats } from '../../dist/neuralese/deltas.js';
import { HttpNeuraleseStore } from '../../dist/model/neuralese-server.js';
import { bounded, caseTarget, casesByFamily, decisionSession, quality } from './decision-lib.mjs';

const NUMERIC = ['generations', 'population', 'support', 'validation', 'query', 'steps', 'lr', 'seed-texts', 'guidance-words', 'reference-steps'];
const options = { generations: 24, population: 6, support: 16, validation: 16, query: 24, steps: 8, lr: 0.02, 'seed-texts': 3,
  'guidance-words': 60, fitness: 'quality', 'reference-steps': 8, init: 'encode', 'author-endpoint': 'http://127.0.0.1:8082', 'author-model': 'nvidia/Qwen3.6-35B-A3B-NVFP4' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (![...NUMERIC, 'cases', 'out', 'endpoint', 'families', 'author-endpoint', 'author-model', 'fitness', 'init'].includes(key) || value === undefined)
    throw Error('Usage: see the header of memetic-decision.mjs');
  options[key] = NUMERIC.includes(key) ? Number(value) : value;
}
if (!options.cases || !options.out || !options.endpoint || !options.families) throw Error('cases, out, endpoint and families are required');
if (!['quality', 'logloss'].includes(options.fitness)) throw Error('--fitness is quality or logloss');
// The method arms' generic soft-init text (run-method-arms.mjs).
const INIT_TEXT = 'Read the input closely, weigh the evidence for each allowed answer, and give the answer the evidence supports.';
const OPERATORS = ['propose', 'embed', 'refine-lamarck', 'refine-baldwin', 'merge'];
const sha = value => createHash('sha256').update(value).digest('hex');

const byFamily = casesByFamily(options.cases);
const families = options.families.split(',');
for (const f of families) {
  const entry = byFamily.get(f);
  if (!entry || entry.train.length < options.support + options.validation || entry.heldout.length < options.query)
    throw Error(`${f}: not enough cases`);
}
const out = resolve(options.out);
await mkdir(join(out, 'artifacts'), { recursive: true });
await writeFile(join(out, 'run.json'), JSON.stringify({ version: 'natlang.memetic-decision/1', options, operators: OPERATORS,
  cases_sha256: sha(readFileSync(options.cases)) }, null, 2) + '\n', { flag: 'wx' });
const session = decisionSession(options.endpoint);
const { learning, readoutOf, lossOn, readouts } = session;
// Crisp → soft: the text encoded in one pass through the port, or with `--init embed` its raw token embeddings (v1, v2).
const embed = text => options.init === 'embed' ? session.embed(text) : session.encode(text);
const remote = new HttpNeuraleseStore(options.endpoint);

// Author ---------------------------------------------------------------------------------------------------------
async function author(prompt, attempt = 0) {
  // The author server can restart under a long run; wait for it rather than lose the run.
  try { return await authorOnce(prompt); }
  catch (error) {
    if (attempt >= 60) throw error;  // about an hour: a model server restart includes loading its weights
    await new Promise(done => setTimeout(done, 60_000));
    return author(prompt, attempt + 1);
  }
}
async function authorOnce(prompt) {
  const response = await fetch(`${options['author-endpoint']}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: options['author-model'], messages: [{ role: 'user', content: prompt }], max_tokens: 2048, temperature: 0.8,
      chat_template_kwargs: { enable_thinking: false } }) });
  if (!response.ok) throw Error(`author HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  const text = (await response.json()).choices[0].message.content ?? '';
  const match = /<guidance>([\s\S]*?)<\/guidance>/.exec(text);
  // A small model is distracted by long guidance: keep at most the asked-for number of words.
  return (match ? match[1] : text).trim().split(/\s+/).slice(0, options['guidance-words']).join(' ');
}
const excerpt = text => String(text).length > 600 ? String(text).slice(0, 600) + '…' : String(text);
function describeCase(c, predicted) {
  const { values, gold } = caseTarget(c);
  return `Input: ${excerpt(c.state)}\nCorrect: ${values.map((v, i) => gold[i] > 0 ? `${JSON.stringify(v)} (${gold[i].toFixed(2)})` : null).filter(Boolean).join(', ')}\n` +
    `Model's probabilities: ${values.map((v, i) => `${JSON.stringify(v)} ${predicted[i].toFixed(2)}`).join(', ')}`;
}
const seedPrompt = (family, question, support) => `You write short guidance for a small language model that answers a classification question.
The question it answers: ${question}
Examples with the correct answers:
${support.slice(0, 6).map(c => describeCase(c, caseTarget(c).gold)).join('\n\n')}

Write guidance (at most ${options['guidance-words']} words) that tells the model how to decide: which cues matter, common confusions, how to weigh
uncertain cases. Do not quote the examples. Reply with the guidance inside <guidance>…</guidance>.`;
const revisePrompt = (question, guidance, worst) => `You improve guidance for a small language model that answers: ${question}
Current guidance:
${guidance || '(none)'}

Cases where the model with this guidance does worst (the share is how much of the training gradient each case drives):
${worst.map(({ c, predicted, share }) => `${describeCase(c, predicted)}\nGradient share: ${(100 * share).toFixed(0)}%`).join('\n\n')}

Write revised guidance (at most ${options['guidance-words']} words) that fixes these errors without overfitting to these inputs. Do not quote them.
Reply with the guidance inside <guidance>…</guidance>.`;

// Individuals ----------------------------------------------------------------------------------------------------
let serial = 0;
const individual = (fields) => ({ id: `ind-${++serial}`, ...fields });
const params = ind => ({ guidance: ind.guidance, skill: ind.skill });
async function evaluateOn(cases, ind) {
  const rows = await bounded(cases, 4, async c => {
    const predicted = await readoutOf(c, params(ind));
    const { gold } = caseTarget(c);
    return { c, predicted, logLoss: -gold.reduce((s, g, i) => s + (g ? g * Math.log(Math.max(predicted[i], 1e-12)) : 0), 0),
      quality: quality(c, predicted, gold) };
  });
  // `perCase` and `loss` are what selection minimises: 1 − quality, or the log loss.
  const cost = r => options.fitness === 'quality' ? 1 - r.quality : r.logLoss;
  const mean = values => values.reduce((s, x) => s + x, 0) / values.length;
  return { perCase: rows.map(cost), loss: mean(rows.map(cost)), logLoss: mean(rows.map(r => r.logLoss)),
    quality: mean(rows.map(r => r.quality)), rows };
}
async function refine(ind, support, steps = options.steps) {
  const adam = learning.optimizers.adam({ lr: options.lr });
  const loss = skill => lossOn(support)({ guidance: ind.guidance, skill });
  const step = async state => { const { grad } = await learning.valueAndGrad(loss, state.value); return adam.step(state, grad); };
  const final = await session.runtime.run(() => iterateOn(step, { value: ind.skill, opt: adam.init(ind.skill) })
    .withLimit({ maxSteps: steps }).checkProgress('off').until(() => false))
    .catch(error => { if (error?.name === 'IterationLimitError') return error.lastState; throw error; });
  return final.value;
}
/** Per-case share of the gradient norm on the support set: which cases drive the update (the gradient digest). */
async function gradientShares(ind, cases) {
  const norms = [];
  for (const c of cases) {
    const { grad } = await learning.valueAndGrad(skill => lossOn([c])({ guidance: ind.guidance, skill }), ind.skill);
    const entry = grad.$gradient;
    const block = entry ? await remote.get(entry.$gradientBlock.id) : undefined;
    norms.push(block ? Math.sqrt(blockFloats(block).reduce((s, x) => s + x * x, 0)) : 0);
  }
  const total = norms.reduce((s, x) => s + x, 0) || 1;
  return norms.map(n => n / total);
}

// Selection ------------------------------------------------------------------------------------------------------
function select(population, size) {
  const front = population.filter(a => !population.some(b => b !== a &&
    b.fit.perCase.every((x, i) => x <= a.fit.perCase[i]) && b.fit.perCase.some((x, i) => x < a.fit.perCase[i])));
  const ranked = [...front].sort((a, b) => a.fit.loss - b.fit.loss);
  for (const ind of [...population].sort((a, b) => a.fit.loss - b.fit.loss)) if (!ranked.includes(ind)) ranked.push(ind);
  return ranked.slice(0, size);
}
const tournament = population => {
  const a = population[Math.floor(Math.random() * population.length)], b = population[Math.floor(Math.random() * population.length)];
  return a.fit.loss <= b.fit.loss ? a : b;
};
class Bandit {
  constructor(arms) { this.stats = Object.fromEntries(arms.map(arm => [arm, { n: 0, reward: 0 }])); this.t = 0; }
  choose(allowed) {
    this.t++;
    const untried = allowed.filter(arm => !this.stats[arm].n);
    if (untried.length) return untried[0];
    return allowed.reduce((best, arm) => this.score(arm) > this.score(best) ? arm : best);
  }
  score(arm) { const s = this.stats[arm]; return s.reward / s.n + Math.sqrt(2 * Math.log(this.t) / s.n); }
  credit(arm, reward) { this.stats[arm].n++; this.stats[arm].reward += reward; }
}

// Run ------------------------------------------------------------------------------------------------------------
const record = entry => appendFile(join(out, 'results.jsonl'), JSON.stringify(entry) + '\n');
const stepRecord = entry => appendFile(join(out, 'improvement-steps.jsonl'), JSON.stringify(improvementStep(entry)) + '\n');
const refs = ind => [{ kind: 'instruction', id: sha(ind.guidance ?? ''), role: 'guidance' },
  ...(ind.skill ? [{ kind: 'soft-skill', id: ind.skill.$neuralese.id, role: 'skill' }] : [])];

for (const [index, family] of families.entries()) {
  const other = families[(index + 1) % families.length];
  const entry = byFamily.get(family);
  const support = entry.train.slice(0, options.support), validation = entry.train.slice(options.support, options.support + options.validation);
  const query = entry.heldout.slice(0, options.query), transfer = byFamily.get(other).heldout.slice(0, options.query);
  const question = support[0].question;
  const bandit = new Bandit(OPERATORS);
  const log = [];
  // Seeds: the bare question with a generic skill text, and author-written guidance texts, each with its embedding.
  let population = [];
  const familyReadouts = readouts();
  const plain = individual({ guidance: '', skill: await embed(INIT_TEXT), parent: null, operator: 'seed' });
  plain.origin = plain.skill.$neuralese.id;
  population.push(plain);
  for (let i = 0; i < options['seed-texts']; i++) {
    const guidance = await author(seedPrompt(family, question, support.slice(i * 3).concat(support).slice(0, 6)));
    const ind = individual({ guidance, skill: await embed(guidance), parent: null, operator: 'seed' });
    ind.origin = ind.skill.$neuralese.id;
    population.push(ind);
  }
  for (const ind of population) ind.fit = await evaluateOn(validation, ind);
  const baseline = population[0].fit;
  for (let generation = 0; generation < options.generations; generation++) {
    const parent = tournament(population);
    const relatives = population.filter(ind => ind !== parent && ind.origin === parent.origin && ind.skill.$neuralese.id !== parent.skill.$neuralese.id);
    const allowed = OPERATORS.filter(op => op !== 'merge' || relatives.length);
    const operator = bandit.choose(allowed);
    const started = Date.now(), callsBefore = readouts();
    let child;
    try {
      if (operator === 'propose') {
        const scored = (await evaluateOn(support, parent)).rows;
        const worstRows = [...scored].sort((a, b) => b.logLoss - a.logLoss).slice(0, 4);
        const shares = await gradientShares(parent, worstRows.map(r => r.c));
        const guidance = await author(revisePrompt(question, parent.guidance, worstRows.map((r, i) => ({ ...r, share: shares[i] }))));
        child = individual({ guidance, skill: parent.skill, origin: parent.origin });
      } else if (operator === 'embed') {
        const skill = await embed(parent.guidance || INIT_TEXT);
        child = individual({ guidance: parent.guidance, skill, origin: skill.$neuralese.id });
      } else if (operator === 'refine-lamarck') {
        child = individual({ guidance: parent.guidance, skill: await refine(parent, support), origin: parent.origin });
      } else if (operator === 'refine-baldwin') {
        const refined = await refine(parent, support);
        child = individual({ guidance: parent.guidance, skill: parent.skill, origin: parent.origin });
        child.fit = await evaluateOn(validation, { guidance: parent.guidance, skill: refined });
        child.baldwin = refined.$neuralese.id;
      } else {
        const mate = relatives[Math.floor(Math.random() * relatives.length)];
        const origin = { $neuralese: { type: 'Neuralese<string>', id: parent.origin } };
        const [d1, d2] = [await learning.deltas.diff(parent.skill, origin), await learning.deltas.diff(mate.skill, origin)];
        const merged = await learning.deltas.learnMerge(origin, [d1, d2], skill => lossOn(support)({ guidance: parent.guidance, skill }),
          { steps: options.steps, lr: 0.2, init: 0.5 });
        child = individual({ guidance: parent.guidance, skill: merged.value, origin: parent.origin, merge: { mate: mate.id, coefficients: merged.coefficients } });
      }
      child.parent = parent.id; child.operator = operator;
      child.fit ??= await evaluateOn(validation, child);
    } catch (error) {
      bandit.credit(operator, 0);
      log.push({ generation, operator, parent: parent.id, error: String(error?.message ?? error).slice(0, 300) });
      continue;
    }
    const seconds = (Date.now() - started) / 1000, gain = parent.fit.loss - child.fit.loss;
    bandit.credit(operator, Math.max(0, gain) / Math.max(1, seconds) * 60);
    population = select([...population, child], options.population);
    const kept = population.includes(child);
    log.push({ generation, operator, parent: parent.id, child: child.id, parent_loss: parent.fit.loss, child_loss: child.fit.loss, kept, seconds });
    console.log(JSON.stringify({ family, generation, operator, gain: +gain.toFixed(4), child_loss: +child.fit.loss.toFixed(4), kept, seconds }));
    await stepRecord({ episode: { id: `memetic:${family}`, family }, facets: [`family:${family}`, `operator:${operator}`, 'search:memetic'],
      before: refs(parent), operator: { kind: `memetic:${operator}`, version: 'memetic-decision/1',
        regime: operator.startsWith('refine') || operator === 'merge' ? 'supervised' : 'search',
        hyper: { steps: options.steps, lr: options.lr, ...(child.merge ? { merge: child.merge } : {}), ...(child.baldwin ? { refined: child.baldwin } : {}) },
        context: null, model: operator === 'propose' ? options['author-model'] : null },
      view: { visibility: 'full', evidence: [] },
      proposal: { deltas: refs(child).filter(ref => !refs(parent).some(p => p.id === ref.id)).map(artifact => ({ artifact, delta: null, scale: 1 })) },
      after: refs(child),
      outcome: { disposition: kept ? 'kept' : 'discarded', support: null,
        query: null, transfer: null,
        compute: { seconds, model_calls: readouts() - callsBefore, operator_requests: operator === 'propose' ? 1 : 0 } },
      trajectory: { id: `memetic:${family}`, step: generation } });
  }
  const searchCompute = { seconds: log.reduce((s, e) => s + (e.seconds ?? 0), 0), readout_calls: readouts() - familyReadouts };
  const best = [...population].sort((a, b) => a.fit.loss - b.fit.loss)[0];
  // Reference on the same split: the soft-gold method arm (refinement of the generic seed alone).
  const referenceStarted = Date.now(), referenceReadouts = readouts();
  const reference = { guidance: '', skill: await refine(plain, support, options['reference-steps']) };
  const referenceCompute = { seconds: (Date.now() - referenceStarted) / 1000, readout_calls: readouts() - referenceReadouts };
  const [bestQuery, bestTransfer, baseQuery, baseTransfer, refQuery, refValidation] = [await evaluateOn(query, best), await evaluateOn(transfer, best),
    await evaluateOn(query, plain), await evaluateOn(transfer, plain), await evaluateOn(query, reference), await evaluateOn(validation, reference)];
  await record({ family, transfer_family: other, fitness: options.fitness, best: { id: best.id, operator: best.operator, guidance: best.guidance,
    skill: best.skill.$neuralese.id, validation_loss: best.fit.loss, validation_quality: best.fit.quality }, seed_validation_loss: baseline.loss,
    query: { best: bestQuery.quality, seed: baseQuery.quality, 'soft-gold': refQuery.quality },
    validation: { best: best.fit.quality, seed: baseline.quality, 'soft-gold': refValidation.quality },
    transfer: { best: bestTransfer.quality, seed: baseTransfer.quality },
    compute: { search: searchCompute, 'soft-gold': referenceCompute }, bandit: bandit.stats, log });
  // The final population's soft skills and the reference, so the steps stay resolvable after the server is gone.
  await learning.save(join(out, 'artifacts', `${family.replace(/[^a-z0-9-]+/gi, '_')}.nz`), Object.fromEntries([...population.map(ind =>
    [`${ind.id.replace('-', '_')}_skill`, { type: 'Neuralese<string>', value: ind.skill }]),
    ['soft_gold_skill', { type: 'Neuralese<string>', value: reference.skill }]]));
  console.log(JSON.stringify({ family, query_best: bestQuery.quality, query_seed: baseQuery.quality, query_soft_gold: refQuery.quality,
    transfer_best: bestTransfer.quality, compute: { search: searchCompute, 'soft-gold': referenceCompute } }));
}
