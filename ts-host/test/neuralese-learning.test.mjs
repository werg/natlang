/**
 * `natlang:learning` and `natlang:neuralese` end to end against the Python reference server with the base LFM2.5-350M
 * weights and untrained port heads. The model is scripted with the server's forced-plan hook, so the tests check the
 * machinery: recording turns, gradient replay, immutable optimiser steps, iteration, saving, the standard library and
 * graph nodes. Skipped when the Neuralese venv or the model is not available.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { Context, createNatlangRuntime, softFunction, iterateOn, decodeNz, buildProject,
  learningService, createLearning, buildStandardLibrary, loadStandardLibrary, createNeuraleseLibrary, Loss, LearningError } from '../dist/index.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { isNeuraleseRef } from '../dist/native/neuralese.js';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';
import { RewriteGate } from '../dist/compiler/rewrites.js';

const repo = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.NATLANG_NEURALESE_PYTHON ?? join(repo, '.venv-neuralese', 'bin', 'python');
const fallbackPython = join(homedir(), 'natlang', '.venv-neuralese', 'bin', 'python');
const executable = existsSync(python) ? python : existsSync(fallbackPython) ? fallbackPython : null;
const model = join(homedir(), '.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M');
const skip = !executable || !existsSync(model) ? 'Neuralese venv or LFM2.5-350M not available' : false;

let server, endpoint;
before(async () => {
  if (skip) return;
  server = spawn(executable, ['-m', 'natlang_neuralese.serve', '--port', '0', '--max-block', '4', '--threads', '8'], {
    cwd: join(repo, 'training', 'neuralese'), env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' },
    stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', chunk => { stderr += chunk; });
  endpoint = await new Promise((resolve, reject) => {
    let out = '';
    server.stdout.on('data', chunk => {
      out += chunk;
      const line = out.split('\n').find(item => item.startsWith('{'));
      if (line) resolve(JSON.parse(line).listening);
    });
    server.on('exit', code => reject(new Error(`server exited (${code}): ${stderr.slice(-2000)}`)));
  });
});
after(() => server?.kill());

const ANSWER = ["<|tool_call_start|>[return_result(status='success', value='Lyon')]<|tool_call_end|>"];

async function embed(text, type = 'Neuralese<string>') {
  const response = await fetch(`${endpoint}/v1/neuralese/embed`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, type }) });
  return (await response.json()).id;
}

test('valueAndGrad + adam under iterateOn tune a Neuralese context item, and the loss moves', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store, request: { x_natlang_forced: ANSWER } });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store } });
  const service = learningService({ endpoint, store });
  const { valueAndGrad, objectives, optimizers, stopGradient, save } = createLearning(service);
  const body = await embed('Answer the question. The hint in scope tells you which city to name.', 'Neuralese<(q: string) => string>');
  const hint0 = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city in France') } };
  // The program: a soft function whose context holds the hint as a data item.
  const ask = hint => softFunction({ type: '(q: string) => string', body, context: Context.empty().with({ hint }) });
  const loss = hint => objectives.crossEntropy(runtime.run(() => ask(hint)('What is the capital of France?')), 'Paris');

  const first = await valueAndGrad(loss, hint0);
  assert.ok(first.loss instanceof Loss && Number.isFinite(+first.loss));
  assert.match(first.grad.$gradient.$gradientBlock.id, /^nz1_/);
  // stopGradient: a constant gets no gradient.
  const constant = await valueAndGrad(loss, stopGradient(hint0));
  assert.equal(constant.grad.$gradient, null);

  const adam = optimizers.adam({ lr: 0.05 });
  const losses = [+first.loss];
  const step = async state => {
    const { loss: value, grad } = await valueAndGrad(loss, state.value);
    losses.push(+value);
    const next = await adam.step(state, grad);
    return { ...next, last: +value };
  };
  const final = await runtime.run(() => iterateOn(step, { value: hint0, opt: adam.init(hint0), last: Infinity })
    .withLimit({ maxSteps: 4 }).checkProgress('off').until(state => state.last < 0)).catch(error => {
    // The limit ends the run: the last checked state is the result.
    if (error?.name === 'IterationLimitError' || /limit of 4 steps/.test(String(error?.message))) return error.lastState;
    throw error;
  });
  assert.notEqual(final.value.$neuralese.id, hint0.$neuralese.id, 'a step returns a new value; the old one is unchanged');
  const after = +(await valueAndGrad(loss, final.value)).loss;
  assert.ok(after < losses[0], `loss should move down: ${losses.join(', ')} → ${after}`);
  // Save the tuned value as a .nz file.
  const path = join(mkdtempSync(join(tmpdir(), 'natlang-learn-')), 'hint.nz');
  await save(path, { hint: final.value });
  const { header } = decodeNz(new Uint8Array(readFileSync(path)));
  assert.equal(header.exports.hint.type, 'Neuralese<string>');
  // Objectives outside grad, and second order, are refused.
  await assert.rejects(() => objectives.crossEntropy(Promise.resolve(1), 1), LearningError);
  await assert.rejects(() => valueAndGrad(loss, hint0, { order: 2 }), /first-order/);
});

test('natlang:learning needs the learning service', async () => {
  const { valueAndGrad } = createLearning();
  await assert.rejects(() => valueAndGrad(async () => new Loss([]), {}), /learning-unavailable/);
});

test('the standard library: combinator bodies from text, typed readout, rewrite pass in builds', { skip, timeout: 600_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const path = join(mkdtempSync(join(tmpdir(), 'natlang-stdlib-')), 'stdlib.nz');
  const { library } = await buildStandardLibrary({ endpoint, store, path });
  const loaded = await loadStandardLibrary(path, new MemoryNeuraleseStore());
  assert.deepEqual(loaded.bodies, library.bodies);
  const lib = createNeuraleseLibrary(library);
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store, request: { x_natlang_forced: ANSWER } });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store } });
  const v = { $neuralese: { type: 'Neuralese<string>', id: await embed('Lyon') } };
  assert.equal(await runtime.run(() => lib.read(v)), 'Lyon');
  assert.ok(isNeuraleseRef(lib.empty()));
  assert.equal(await lib.combine(v), v, 'combine of one value is that value');
  // Rewrites run in builds and stay off without measurements.
  const root = mkdtempSync(join(tmpdir(), 'natlang-rw-'));
  const { writeFileSync } = await import('node:fs');
  writeFileSync(join(root, 'app.ts'), "import { map, read } from 'natlang:neuralese';\n" +
    "export async function f(v: Neuralese<string>, g: (s: string) => Promise<string>, h: (s: string) => Promise<string>) {\n" +
    '  return read(await map(await map(v, g), h));\n}\n');
  writeFileSync(join(root, 'tsconfig.json'), JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true } }));
  const built = buildProject({ project: root, outDir: join(root, 'dist'), write: false,
    rewriteGate: new RewriteGate({ model: 'lfm', dialect: 'nd:natlang@1' }) });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.deepEqual(built.rewrites, {});
});

test('law objectives: the right side is the readout target of the left, and gradients flow', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const path = join(mkdtempSync(join(tmpdir(), 'natlang-law-')), 'stdlib.nz');
  const { library } = await buildStandardLibrary({ endpoint, store, path });
  const WRITE = ["<|tool_call_start|>[eval(code='const out: Neuralese<string> = ", { neuralese: 'write' },
    ";\\nreturn out;')]<|tool_call_end|>"];
  const RETURN = ["<|tool_call_start|>[return_result(status='success')]<|tool_call_end|>"];
  const turn = plan => neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store, request: { x_natlang_forced: plan } });
  const [answer, write, finish] = [turn(ANSWER), turn(WRITE), turn(RETURN)];
  // Readouts answer in text; combinator calls write a block in eval code, then return it. Requests end with the
  // call opening (a synthetic scope_ eval call and its result), so the step is told apart by a model eval call.
  const driver = Object.assign((request, signal) => {
    const text = JSON.stringify(request);
    if (!text.includes('Neuralese<unknown>')) return answer(request, signal);
    const wrote = request.messages.some(m => m.role === 'assistant' && (m.tool_calls ?? []).some(c => c.function?.name === 'eval' && !String(c.id).startsWith('scope_')));
    return (wrote ? finish : write)(request, signal);
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store } });
  const { valueAndGrad, objectives } = createLearning(learningService({ endpoint, store }), { library });
  const v = { $neuralese: { type: 'Neuralese<string>', id: await embed('Lyon') } };
  const { loss, grad } = await valueAndGrad(value => runtime.run(() => objectives.law('combineIdentity', value)), v);
  assert.equal(loss.terms.length, 1);
  assert.equal(loss.terms[0].law, 'combineIdentity');
  assert.match(loss.terms[0].target.tool_calls[0].function.arguments, /Lyon/, 'the right side read(v) is the target');
  assert.ok(Number.isFinite(+loss));
  assert.ok('$gradient' in grad);
  await assert.rejects(() => valueAndGrad(value => runtime.run(() => objectives.law('mapIdentity', value, 42)), v),
    /must be a function/);
  await assert.rejects(() => valueAndGrad(value => runtime.run(() => objectives.law('noSuchLaw', value)), v), /unknown law/);
  // Without a library the law says what it needs.
  const bare = createLearning(learningService({ endpoint, store }));
  await assert.rejects(() => bare.valueAndGrad(value => runtime.run(() => bare.objectives.law('combineIdentity', value)), v),
    /standard library/);
});

test('a decision readout trains with a proper scoring rule: objectives.decision moves its distribution', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store },
    trace: trace => traces.push(trace) });
  const { valueAndGrad, objectives, optimizers } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with the city the hint names.', 'Neuralese<(q: string) => string>');
  const hint0 = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city in France') } };
  const ask = hint => softFunction({ type: '(q: string) => "Paris" | "Lyon" | "Rome"', body, context: Context.empty().with({ hint }) });
  const target = { Paris: 0.1, Lyon: 0.8, Rome: 0.1 };
  const loss = hint => objectives.decision(runtime.run(() => ask(hint)('Which city?')), target, 'logLoss');
  const first = await valueAndGrad(loss, hint0);
  assert.ok(Number.isFinite(+first.loss) && first.grad.$gradient);
  const scored = traces.flatMap(trace => trace.events).find(event => JSON.stringify(event).includes('"decision_readout"') && event.probabilities);
  assert.equal(scored?.probabilities.length, 3);
  const adam = optimizers.adam({ lr: 0.1 });
  let state = { value: hint0, opt: adam.init(hint0) }, grad = first.grad;
  for (let i = 0; i < 3; i++) {
    state = await adam.step(state, grad);
    grad = (await valueAndGrad(loss, state.value)).grad;
  }
  const after = +(await valueAndGrad(loss, state.value)).loss;
  assert.ok(after < +first.loss, `decision loss should move down: ${+first.loss} → ${after}`);
});

test('concurrent objectives: functions keep their own turns, overlapping promises are refused', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store, request: { x_natlang_forced: ANSWER } });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store } });
  const { valueAndGrad, objectives } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question.', 'Neuralese<(q: string) => string>');
  const hint = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city') } };
  const ask = h => softFunction({ type: '(q: string) => string', body, context: Context.empty().with({ hint: h }) });
  const questions = ['Capital of France?', 'Capital of Italy?', 'Capital of Spain?'];
  let terms;
  await valueAndGrad(async h => {
    const losses = await Promise.all(questions.map(q => objectives.crossEntropy(() => runtime.run(() => ask(h)(q)), 'Paris')));
    terms = losses.map(loss => loss.terms[0]);
    return objectives.sum(...losses);
  }, hint);
  // Each term carries its own question, whatever order the turns finished in.
  for (const [index, term] of terms.entries()) assert.match(JSON.stringify(term.messages), new RegExp(questions[index].replace('?', '\\?')));
  await assert.rejects(() => valueAndGrad(async h => objectives.sum(...await Promise.all(questions.map(q =>
    objectives.crossEntropy(runtime.run(() => ask(h)(q)), 'Paris')))), hint), /learning-concurrent-objectives/);
});

test('a weight adapter is a value: withAdapters binds it to calls, valueAndGrad tunes it, and its type is Adapter', { skip, timeout: 900_000 }, async () => {
  const { parseType, formatType } = await import('../dist/native/types.js');
  assert.equal(formatType(parseType('Adapter')), 'Adapter');
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store } });
  const { valueAndGrad, objectives, optimizers, adapters, withAdapters, save, deltas } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with a city.', 'Neuralese<(q: string) => string>');
  const ask = softFunction({ type: '(q: string) => "Paris" | "Lyon" | "Rome"', body });
  const adapter0 = await adapters.create({ kind: 'xs', rank: 4 });
  assert.equal(adapter0.$neuralese.type, 'Adapter');
  const target = { Paris: 0.1, Lyon: 0.8, Rome: 0.1 };
  let terms;
  const loss = async adapter => {
    const value = await objectives.decision(() => withAdapters(adapter, () => runtime.run(() => ask('Which city?'))), target);
    terms = value.terms;
    return value;
  };
  const first = await valueAndGrad(loss, adapter0);
  assert.deepEqual(terms[0].adapters, [{ id: adapter0.$neuralese.id, scale: 1 }]);
  assert.match(first.grad.$gradient.$gradientBlock.id, /^nz1_/);
  const adam = optimizers.adam({ lr: 0.02 });
  let state = { value: adapter0, opt: adam.init(adapter0) }, grad = first.grad;
  for (let i = 0; i < 3; i++) {
    state = await adam.step(state, grad);
    grad = (await valueAndGrad(loss, state.value)).grad;
  }
  assert.equal(state.value.$neuralese.type, 'Adapter');
  const after = +(await valueAndGrad(loss, state.value)).loss;
  assert.ok(after < +first.loss, `adapter loss should move down: ${+first.loss} → ${after}`);
  // Outside the scope the base model answers: the zero adapter and no adapter score alike.
  const base = +(await valueAndGrad(async () => objectives.decision(() => runtime.run(() => ask('Which city?')), target), {})).loss;
  assert.ok(Math.abs(base - +first.loss) < 1e-3, `zero adapter ${+first.loss} vs base ${base}`);
  // An adapter ships in a .nz file: its block keeps its own dialect (the spec), and it loads back as an Adapter.
  const path = join(mkdtempSync(join(tmpdir(), 'natlang-adapter-')), 'adapter.nz');
  await save(path, { adapter: state.value });
  const decoded = decodeNz(new Uint8Array(readFileSync(path)));
  assert.equal(decoded.header.exports.adapter.type, 'Adapter');
  assert.match(decoded.blocks.get(state.value.$neuralese.id).meta.dialect, /^adapter\/1;base=/);
  // The training run as a residual update: diff(tuned, zero) applied to zero is the tuned adapter; a delta is not
  // an adapter and cannot be bound.
  const update = await deltas.diff(state.value, adapter0);
  assert.equal((await deltas.apply(adapter0, update)).$neuralese.id, state.value.$neuralese.id);
  const half = await deltas.apply(adapter0, update, 0.5);
  assert.ok(Number.isFinite(+(await valueAndGrad(loss, half)).loss), "a half-scale update is an adapter that runs");
  await assert.rejects(() => valueAndGrad(loss, update), /neuralese-adapter|not an adapter/);
});

test('conditionedDistill: a soft hint learns the readout of a teacher told the answer; leaks and unconditioned teachers are refused', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store } });
  const { valueAndGrad, objectives, optimizers } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with the city the notes in scope name.', 'Neuralese<(q: string) => string>');
  const hint0 = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city in France') } };
  const type = '(q: string) => "Paris" | "Lyon" | "Rome"';
  const secret = 'The city to name is Lyon, not Paris.';
  const student = hint => () => runtime.run(() => softFunction({ type, body, context: Context.empty().with({ hint }) })('Which city?'));
  const teacher = () => runtime.run(() => softFunction({ type, body, context: Context.empty().with({ note: secret }) })('Which city?'));
  let terms;
  const loss = async hint => {
    const value = await objectives.conditionedDistill(student(hint), teacher, { privileged: secret });
    terms = value.terms;
    return value;
  };
  const first = await valueAndGrad(loss, hint0);
  assert.equal(terms[0].kind, 'decision');
  assert.match(JSON.stringify(terms[0].teacher.messages), /Lyon, not Paris/);
  assert.doesNotMatch(JSON.stringify(terms[0].messages), /Lyon, not Paris/);
  const adam = optimizers.adam({ lr: 0.1 });
  let state = { value: hint0, opt: adam.init(hint0) }, grad = first.grad;
  for (let i = 0; i < 3; i++) {
    state = await adam.step(state, grad);
    grad = (await valueAndGrad(loss, state.value)).grad;
  }
  const after = +(await valueAndGrad(loss, state.value)).loss;
  assert.ok(after < +first.loss, `conditioned distillation loss should move down: ${+first.loss} → ${after}`);
  // The privilege check: the student may not see the privileged text.
  const leaky = () => runtime.run(() => softFunction({ type, body, context: Context.empty().with({ hint: hint0, note: secret }) })('Which city?'));
  await assert.rejects(() => valueAndGrad(async () => objectives.conditionedDistill(leaky, teacher, { privileged: secret }), {}), /learning-privilege-leak/);
  // A teacher that saw exactly the student's view conditions on nothing.
  await assert.rejects(() => valueAndGrad(async () => objectives.conditionedDistill(student(hint0), student(hint0)), {}), /learning-teacher-unconditioned/);
});

test('reinforcement: expectedReward moves a decision toward rewarded values; policyGradient weights trajectories by advantage', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store } });
  const { valueAndGrad, objectives, optimizers, trajectory } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with a city.', 'Neuralese<(q: string) => string>');
  const hint0 = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city in France') } };
  const ask = hint => softFunction({ type: '(q: string) => "Paris" | "Lyon" | "Rome"', body, context: Context.empty().with({ hint }) });
  const reward = value => value === 'Lyon' ? 1 : 0;
  let terms;
  const loss = async hint => { const value = await objectives.expectedReward(() => runtime.run(() => ask(hint)('Which city?')), reward); terms = value.terms; return value; };
  const first = await valueAndGrad(loss, hint0);
  assert.equal(terms[0].rule, 'expectedReward');
  assert.ok(Math.abs(terms[0].target.rewards.reduce((a, b) => a + b, 0)) < 1e-12, 'rewards are centred');
  const adam = optimizers.adam({ lr: 0.1 });
  let state = { value: hint0, opt: adam.init(hint0) }, grad = first.grad;
  for (let i = 0; i < 3; i++) { state = await adam.step(state, grad); grad = (await valueAndGrad(loss, state.value)).grad; }
  const after = +(await valueAndGrad(loss, state.value)).loss;
  assert.ok(after < +first.loss, `expected reward should rise: loss ${+first.loss} → ${after}`);

  // Policy gradient over two scripted trajectories with different rewards: weights are ±advantage / n.
  const scripted = answer => neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store,
    request: { x_natlang_forced: [`<|tool_call_start|>[return_result(status='success', value='${answer}')]<|tool_call_end|>`] } });
  const textRuntime = answer => createNatlangRuntime({ model: scripted(answer), neuralese: { store } });
  const plain = hint => softFunction({ type: '(q: string) => string', body, context: Context.empty().with({ hint }) });
  let pgTerms;
  const pg = await valueAndGrad(async hint => {
    const samples = [];
    for (const [answer, r] of [['Lyon', 1], ['Paris', 0]])
      samples.push({ trajectory: await trajectory(() => textRuntime(answer).run(() => plain(hint)('Which city?'))), reward: r });
    const value = await objectives.policyGradient(samples);
    pgTerms = value.terms;
    return value;
  }, hint0);
  assert.deepEqual(pgTerms.map(term => term.weight), [0.25, -0.25]);
  assert.match(pg.grad.$gradient.$gradientBlock.id, /^nz1_/);
  // Tied rewards: no terms, zero loss, no gradient, no server call.
  const tied = await valueAndGrad(async hint => objectives.policyGradient([{ trajectory: { id: 't', turns: [] }, reward: 1 }]), hint0);
  assert.equal(+tied.loss, 0);
  assert.equal(tied.grad.$gradient, null);
});

test('deltas: interference of separately trained updates, and learned merge coefficients lower the loss', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store } });
  const { valueAndGrad, objectives, optimizers, deltas } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with the city the hint names.', 'Neuralese<(q: string) => string>');
  const base = { $neuralese: { type: 'Neuralese<string>', id: await embed('a city in France') } };
  const ask = hint => softFunction({ type: '(q: string) => "Paris" | "Lyon" | "Rome"', body, context: Context.empty().with({ hint }) });
  const lossFor = target => hint => objectives.decision(() => runtime.run(() => ask(hint)('Which city?')), target);
  const train = async target => {
    const adam = optimizers.adam({ lr: 0.1 });
    let state = { value: base, opt: adam.init(base) };
    for (let i = 0; i < 2; i++) state = await adam.step(state, (await valueAndGrad(lossFor(target), state.value)).grad);
    return deltas.diff(state.value, base);
  };
  const toLyon = await train({ Lyon: 1 }), toRome = await train({ Rome: 1 });
  const lyonLoss = lossFor({ Lyon: 0.8, Rome: 0.2 });
  const measure = async value => +(await valueAndGrad(lyonLoss, value)).loss;
  const report = await deltas.interference(base, [toLyon, toRome], measure);
  assert.equal(report.alone.length, 2);
  assert.ok([report.base, report.joint, report.interference].every(Number.isFinite));
  const merged = await deltas.learnMerge(base, [toLyon, toRome], lyonLoss, { steps: 4, lr: 0.2, init: 0.5 });
  assert.equal(merged.trace.length, 4);
  assert.notDeepEqual(merged.coefficients, [0.5, 0.5]);
  assert.ok(merged.trace.at(-1).loss < merged.trace[0].loss, JSON.stringify(merged.trace));
});

test('an adapter bound in a function\'s context applies to that function\'s turns only and is not shown in its scope', { skip, timeout: 900_000 }, async () => {
  const store = new MemoryNeuraleseStore();
  const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store });
  const runtime = createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, neuralese: { store } });
  const { valueAndGrad, objectives, optimizers, adapters } = createLearning(learningService({ endpoint, store }));
  const body = await embed('Answer the question with a city.', 'Neuralese<(q: string) => string>');
  const type = '(q: string) => "Paris" | "Lyon" | "Rome"';
  const bound = adapter => softFunction({ type, body, context: Context.empty().with({ adapter }) });
  const plain = softFunction({ type, body });
  const target = { Paris: 0.1, Lyon: 0.8, Rome: 0.1 };
  let terms;
  const loss = async adapter => { const value = await objectives.decision(() => runtime.run(() => bound(adapter)('Which city?')), target); terms = value.terms; return value; };
  const adapter0 = await adapters.create({ kind: 'xs', rank: 4 });
  const first = await valueAndGrad(loss, adapter0);
  assert.deepEqual(terms[0].adapters, [{ id: adapter0.$neuralese.id, scale: 1 }]);
  assert.doesNotMatch(JSON.stringify(terms[0].messages), new RegExp(adapter0.$neuralese.id), 'the adapter is not rendered into the scope');
  const adam = optimizers.adam({ lr: 0.05 });
  let state = { value: adapter0, opt: adam.init(adapter0) }, grad = first.grad;
  for (let i = 0; i < 3; i++) { state = await adam.step(state, grad); grad = (await valueAndGrad(loss, state.value)).grad; }
  const after = +(await valueAndGrad(loss, state.value)).loss;
  assert.ok(after < +first.loss, `context-bound adapter loss should move down: ${+first.loss} → ${after}`);
  const unbound = +(await valueAndGrad(async () => objectives.decision(() => runtime.run(() => plain('Which city?')), target), {})).loss;
  assert.ok(Math.abs(unbound - +first.loss) < 1e-3, 'a function that does not bind the adapter runs the base model');
});
