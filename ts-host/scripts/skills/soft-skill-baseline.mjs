#!/usr/bin/env node
/** Soft-skill gradient baseline on graded skill episodes (S6 gradient-tuning baseline).
 *
 * For each episode the target function gets a `skill: Neuralese<string>` data item in its context, which its call
 * opening lists as a literal. Arms, all scored on the SEALED query cases by the negative log-likelihood of the gold
 * answer as the call's result:
 *   none       — no skill item;
 *   text-init  — the skill block initialised from the token embeddings of a crisp skill text;
 *   tuned      — that block after K Adam steps of valueAndGrad on support-case cross-entropy only.
 * Support supervision forces the gold `return_result` so that each case records exactly one turn (its opening).
 * Query cases are never part of any gradient. Each arm is also scored on the episode's transfer cases (another family),
 * which separates a family skill from generic answer-format priming, and with `--sample` every arm answers the query
 * and transfer cases freely, scored by the episode's graded metric. Results keep every arm and step. */
import { mkdtempSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { Context, createNatlangRuntime, iterateOn, learningService, createLearning } from '../../dist/index.js';
import { loadNatlang } from '../../dist/runtime/node.js';
import { MemoryNeuraleseStore } from '../../dist/native/neuralese-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { validateEpisode } from '../../dist/skills/episode.js';
import { episodeScorings } from '../../dist/skills/scoring.js';

const options = { limit: 4, steps: 8, lr: 0.02, split: 'train', sample: 'false' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (!['episodes', 'out', 'endpoint', 'limit', 'steps', 'lr', 'init-text', 'split', 'sample', 'database-root'].includes(key) || value === undefined)
    throw Error('Usage: soft-skill-baseline.mjs --episodes FILE --out DIR --endpoint URL [--limit N --steps K --lr X --init-text FILE --sample true --database-root DIR]');
  options[key] = ['limit', 'steps', 'lr'].includes(key) ? Number(value) : value;
}
if (!options.episodes || !options.out || !options.endpoint) throw Error('episodes, out and endpoint are required');
const sha = value => createHash('sha256').update(value).digest('hex');
const initText = options['init-text'] ? readFileSync(options['init-text'], 'utf8') :
  'Read the task carefully. Work out the answer step by step, check it against every requirement, and return exactly the requested form.';

/** The gold answer that supervision and evaluation score. Episodes without one are skipped, not guessed. */
function goldAnswer(row) {
  const expected = row.expected;
  if (expected && expected.kind === 'sql-gold') return expected.sql;
  if (expected && expected.kind === 'gold-answer') return [expected.value].flat()[0];
  if (expected && expected.kind === 'assignment') return expected.value;
  if (expected && expected.kind === 'function-calls') return expected.calls;
  if (expected && expected.kind === 'relevant-set') return expected.items;
  return undefined;
}

async function embed(text, type) {
  const response = await fetch(`${options.endpoint}/v1/neuralese/embed`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, type }) });
  if (!response.ok) throw Error(`embed failed: ${response.status} ${await response.text()}`);
  return (await response.json()).id;
}

function loadTarget(episode) {
  const root = mkdtempSync(join(tmpdir(), 'soft-skill-target-'));
  for (const [path, text] of Object.entries(episode.target.files)) {
    mkdirSync(dirname(join(root, path)), { recursive: true }); writeFileSync(join(root, path), text);
  }
  return loadNatlang(join(root, episode.target.entry), root);
}

const rows = readFileSync(options.episodes, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line))
  .filter(row => row.split === options.split).slice(0, options.limit);
const out = resolve(options.out);
await mkdir(out, { recursive: true });
await writeFile(join(out, 'baseline.json'), JSON.stringify({ version: 'natlang.soft-skill-baseline/2', options,
  input_sha256: sha(readFileSync(options.episodes)), init_text_sha256: sha(initText), episodes: rows.map(row => row.id) }, null, 2) + '\n', { flag: 'wx' });

for (const episode of rows) {
  if (validateEpisode(episode).length) throw Error('invalid episode ' + episode.id);
  const support = episode.support.cases.filter(row => goldAnswer(row) !== undefined);
  const query = episode.query.cases.filter(row => goldAnswer(row) !== undefined);
  const transfer = (episode.transfer?.cases ?? []).filter(row => goldAnswer(row) !== undefined);
  if (!support.length || !query.length) { console.log(JSON.stringify({ episode: episode.id, skipped: 'no gold answers' })); continue; }
  const store = new MemoryNeuraleseStore();
  const solve = loadTarget(episode);
  const base = Context.ofCallable(solve);
  const service = learningService({ endpoint: options.endpoint, store });
  const { valueAndGrad, objectives, optimizers } = createLearning(service);
  // Each case runs with the gold answer forced, so its single recorded turn is the call opening.
  const runCase = (skill, row) => {
    const forced = `<|tool_call_start|>[return_result(status='success', value=${JSON.stringify(goldAnswer(row))})]<|tool_call_end|>`;
    const driver = neuraleseServerModelTurn({ endpoint: options.endpoint, model: 'natlang-neuralese', store, request: { x_natlang_forced: [forced] } });
    const runtime = createNatlangRuntime({ model: driver, neuralese: { store } });
    const bound = skill ? solve.in(base.with({ skill })) : solve;
    return objectives.crossEntropy(runtime.run(() => bound(...row.args)), goldAnswer(row));
  };
  const lossOn = cases => async skill => objectives.sum(...await Promise.all(cases.map(row => runCase(skill, row))));
  const value = async (cases, skill) => +(await valueAndGrad(lossOn(cases), skill ?? null)).loss / cases.length;
  const { scoring, transferScoring } = episodeScorings(episode.provenance, !!episode.transfer, { pins: {}, databaseRoot: options['database-root'] });
  // Free answers: the model writes its own result; the host scores it with the episode's metric.
  const sampled = async (cases, skill, score) => {
    if (options.sample !== 'true' || !score || !cases.length) return undefined;
    const runtime = createNatlangRuntime({ model: neuraleseServerModelTurn({ endpoint: options.endpoint, model: 'natlang-neuralese', store }), neuralese: { store } });
    const bound = skill ? solve.in(base.with({ skill })) : solve;
    let total = 0;
    for (const row of cases) {
      const output = await runtime.run(() => bound(...row.args)).then(value => ({ value }), error => ({ error: String(error?.message ?? error) }));
      total += score.score(row, output).quality;
    }
    return total / cases.length;
  };
  const arm = async skill => ({ query_nll: await value(query, skill), transfer_nll: transfer.length ? await value(transfer, skill) : null,
    query_quality: await sampled(query, skill, scoring), transfer_quality: await sampled(transfer, skill, transferScoring) });
  const record = async entry => appendFile(join(out, 'results.jsonl'), JSON.stringify({ episode: episode.id, family: episode.family, ...entry }) + '\n');

  const none = await arm(undefined);
  await record({ arm: 'none', ...none });
  const init = { $neuralese: { type: 'Neuralese<string>', id: await embed(initText, 'Neuralese<string>') } };
  const initArm = await arm(init);
  await record({ arm: 'text-init', ...initArm, support_nll: await value(support, init) });
  const adam = optimizers.adam({ lr: options.lr });
  const trace = [];
  const step = async state => {
    const { loss, grad } = await valueAndGrad(lossOn(support), state.value);
    trace.push(+loss / support.length);
    return adam.step(state, grad);
  };
  // The step limit ends the run; its last checked state is the tuned value.
  const loop = createNatlangRuntime({ model: neuraleseServerModelTurn({ endpoint: options.endpoint, model: 'natlang-neuralese', store }), neuralese: { store } });
  const tuned = await loop.run(() => iterateOn(step, { value: init, opt: adam.init(init) }).withLimit({ maxSteps: options.steps })
    .checkProgress('off').until(() => false)).catch(error => {
    if (error?.name === 'IterationLimitError') return error.lastState;
    throw error;
  });
  const tunedArm = await arm(tuned.value);
  await record({ arm: 'tuned', ...tunedArm, support_trace: trace, block: tuned.value?.$neuralese?.id ?? null });
  console.log(JSON.stringify({ episode: episode.id, none, text_init: initArm, tuned: tunedArm, support_first: trace[0], support_last: trace.at(-1) }));
}
