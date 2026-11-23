/**
 * Runnable forms of the `natlang:learning` and iteration cases in conformance/neuralese/cases/termination-learning-graph.yaml.
 *
 * Learning cases run against a real Python native engine (tests/neuralese/learning_conformance_server.py): every
 * forward, native-state write and import, recorded turn, replay and gradient is the engine's own. Only token selection
 * is a fixture policy there, and the driver below turns each reply into a typed `return_result` (the tiny model cannot
 * write tool calls). Iteration cases need no server: their model is a recording mock.
 *
 * Each export takes the shared environment and throws on a failed expectation.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as natlang from '../../dist/index.js';
import { compileVirtualProject, createNatlangRuntime, defineNatlang, iterateOn, softFunction, Context } from '../../dist/index.js';
import { modelTurnsSoFar } from '../../dist/native/agent.js';
import { IterationLimitError } from '../../dist/runtime/iterate.js';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { nativeStateConsumerFromInfo, neuraleseRef } from '../../dist/native/neuralese.js';
import { createLearning, learningService, stopGradient } from '../../dist/neuralese/learning.js';
import { replayRecordSink } from '../../dist/neuralese/replay-records.js';

/** The learning environment over a started conformance server: `{ endpoint, identity, target, command }`. */
export async function learningEnvironment(server) {
  const { endpoint } = server;
  const info = await fetch(endpoint + '/v1/neuralese/info').then(response => response.json());
  const dir = mkdtempSync(join(tmpdir(), 'learning-conformance-'));
  const store = await FileNeuraleseStore.open(join(dir, 'archive'), { residentBytes: 0 });
  const records = replayRecordSink({ path: join(dir, 'replay.jsonl'), endpoint });
  const service = learningService({ endpoint, store, replayRecords: records, dialect: info.dialects[0] });
  const learning = createLearning(service);
  const neuralese = { store, dialect: info.dialects[0], nativeStateConsumer: nativeStateConsumerFromInfo(info) };
  const model = neuraleseServerModelTurn({ endpoint, model: 'learning-conformance', store });
  // The reply's exact wire parts become the typed result: a native span stays the actual produced state.
  // `scripts`: [pattern, code] pairs; a call whose opening matches runs `code` in eval on its first turn (after the
  // engine's actual forward), so a recorded trajectory can hold an effect.
  const scripts = [];
  const driver = Object.assign(async request => {
    const raw = await model({ ...request, max_tokens: 8 });
    const opening = String(request.messages.find(message => message.role === 'user')?.content ?? '');
    const script = scripts.find(([pattern]) => pattern.test(opening));
    if (script && modelTurnsSoFar(request.messages) === 0) return { ...raw, text: undefined, calls: [['eval', { code: script[1] }]] };
    const content = raw.raw_response?.choices?.[0]?.message?.content;
    const produced = Array.isArray(content) && content.some(part => part.type === 'native_state');
    return { ...raw, text: undefined, calls: [['return_result', { status: 'success', value: produced ? content : raw.text || 'consumed' }]] };
  }, { neuralese: true, decide: model.decide });
  const effects = [];
  const runtime = createNatlangRuntime({ model: driver, neuralese,
    services: { learning: service, mail: { send: async text => { effects.push(text); return `sent ${effects.length}`; } } } });
  /** A native state written by a real producer call: `() => Neuralese<string>`. */
  const produce = async instructions => {
    const writer = defineNatlang(`---\nreturns: Neuralese<string>\n---\n${instructions}\n`, { name: 'writeState' });
    const value = await runtime.run(() => writer());
    assert.ok(value?.$neuralese?.id?.startsWith('nzs1_'), `the producer wrote no native state: ${JSON.stringify(value)}`);
    return value;
  };
  return { ...server, info, dir, store, service, learning, records, runtime, effects, produce, scripts,
    recordsFile: join(dir, 'replay.jsonl') };
}

const replayRecords = env => {
  try { return readFileSync(env.recordsFile, 'utf8').trim().split('\n').filter(Boolean).map(line => JSON.parse(line)); }
  catch { return []; }
};

const gradientArtifact = async (env, entry, kind = 'state-gradient') => {
  assert.ok(entry?.$gradientBlock, `no gradient entry: ${JSON.stringify(entry)}`);
  const artifact = await env.store.getNativeTensorArtifact(entry.$gradientBlock.id);
  assert.equal(artifact?.metadata.kind, kind);
  return artifact;
};

const values = artifact => Object.values(artifact.tensors).flatMap(tensor => {
  assert.equal(tensor.dtype, 'F32');
  return [...new Float32Array(tensor.data.buffer.slice(tensor.data.byteOffset, tensor.data.byteOffset + tensor.data.byteLength))];
});
const norm = xs => Math.sqrt(xs.reduce((sum, x) => sum + x * x, 0));
const close = (a, b, what) => {
  assert.equal(a.length, b.length, `${what}: shapes differ`);
  const scale = Math.max(norm(a), norm(b), 1e-12);
  const diff = norm(a.map((x, i) => x - b[i]));
  assert.ok(diff <= 1e-4 * scale + 1e-7, `${what}: |a-b| = ${diff} against |a| = ${norm(a)}, |b| = ${norm(b)}`);
};

export const cases = {
  /** valueAndGrad(s => objective(program.in(ctx.with({ skill: s })), cases), ctx.skill) */
  async 'grad-context-item'(env) {
    const program = defineNatlang('---\nargs:\n  ticket: string\nreturns: string\n---\nAnswer the ticket using the skill.\n',
      { name: 'answer' });
    const skill = await env.produce('Write a skill for answering tickets.');
    const ctx = Context.ofCallable(program).with({ skill });
    const cases = [{ ticket: 'refund', expected: 'consumed' }, { ticket: 'login', expected: 'consumed' }];
    const objective = async (fn, examples) => env.learning.objectives.sum(...await Promise.all(examples.map(example =>
      env.learning.objectives.crossEntropy(() => env.runtime.run(() => fn(example.ticket)), example.expected))));
    const before = await env.command('weights');
    const recordsBefore = replayRecords(env).length;
    const result = await env.learning.valueAndGrad(s => objective(program.in(ctx.with({ skill: s })), cases), ctx.data.skill);
    assert.ok(Number.isFinite(+result.loss) && +result.loss > 0, `loss ${+result.loss}`);
    const entry = result.grad.$gradient;
    assert.equal(entry.$gradientBlock.of, skill.$neuralese.id, 'the gradient is bound to the context item');
    assert.equal(entry.$gradientBlock.type, 'NativeState');
    const gradient = await gradientArtifact(env, entry);
    const state = await env.store.getNativeState(skill.$neuralese.id);
    assert.equal(gradient.metadata.source_artifact_id, skill.$neuralese.id);
    // The shape of ctx.skill: a gradient for every layer tensor of the state (its KV and hybrid state), each with that
    // tensor's shape. Token IDs, positions and token embeddings are the span's discrete record, not roots.
    const shapes = tensors => Object.fromEntries(Object.entries(tensors).filter(([name]) => name.startsWith('layers.'))
      .map(([name, tensor]) => [name, tensor.shape]));
    assert.ok(Object.keys(shapes(state.tensors)).length, 'the state has layer tensors');
    assert.deepEqual(shapes(gradient.tensors), shapes(state.tensors), 'the gradient has the state\'s layer tensor shapes');
    assert.deepEqual(Object.keys(gradient.tensors).filter(name => !name.startsWith('layers.')), [], 'no gradient for discrete span records');
    assert.ok(norm(values(gradient)) > 0, 'the gradient is nonzero');
    const records = replayRecords(env).slice(recordsBefore);
    assert.equal(records.length, 1, 'one gradient session record');
    assert.ok(records[0].native_graph_nodes.some(node => node.kind === 'block_read' && node.block === skill.$neuralese.id),
      'the recorded graph reads the context item');
    assert.equal(await env.command('weights'), before, 'model weights unchanged');
  },
};

/** A one-call loss over a context: `answer(ticket)` bound to `items`, scored by cross-entropy of 'consumed'. */
const contextLoss = (env, items) => {
  const program = defineNatlang('---\nargs:\n  ticket: string\nreturns: string\n---\nAnswer the ticket using the skill and the notes.\n',
    { name: 'answer' });
  return env.learning.objectives.crossEntropy(() => env.runtime.run(() => program.in(Context.ofCallable(program).with(items))('refund')), 'consumed');
};

cases['grad-stop-gradient'] = async env => {
  const v = await env.produce('Write a skill for answering tickets.');
  const gradient = async uses => values(await gradientArtifact(env, (await env.learning.grad(s => contextLoss(env, uses(s)), v)).$gradient));
  // Each use of v contributes its own term; stopGradient removes exactly that use's term.
  const first = await gradient(s => ({ skill: s, notes: stopGradient(s) }));
  const second = await gradient(s => ({ skill: stopGradient(s), notes: s }));
  const both = await gradient(s => ({ skill: s, notes: s }));
  assert.ok(norm(first) > 0 && norm(second) > 0, 'each unstopped use has a gradient');
  close(first.map((x, i) => x + second[i]), both, 'the per-use gradients add up to the gradient through both uses');
  assert.ok(norm(first.map((x, i) => x - both[i])) > 1e-6 * norm(both), 'the stopped use contributes nothing');
  await assert.rejects(() => env.learning.grad(s => contextLoss(env, { skill: stopGradient(s), notes: stopGradient(s) }), stopGradient(v)),
    error => error.code === 'learning-native-target', 'a fully stopped argument has no differentiable target');
};

cases['grad-nested-first-order'] = async env => {
  const v = await env.produce('Write a skill for answering tickets.');
  const loss = s => contextLoss(env, { skill: s });
  const sgd = env.learning.optimizers.sgd({ lr: 0.5 });
  let stepped;
  const outer = order => env.learning.grad(async s => {
    const inner = await env.learning.grad(loss, s);
    stepped = (await sgd.step({ value: s, opt: sgd.init(s) }, inner)).value;
    return loss(stepped);
  }, v, order ? { order } : {});
  const firstOrder = values(await gradientArtifact(env, (await outer()).$gradient));
  // With the inner gradient constant, s' = s - lr·g has Jacobian I: the outer gradient is ∇L at s'.
  const atStepped = values(await gradientArtifact(env, (await env.learning.grad(loss, stepped)).$gradient));
  close(firstOrder, atStepped, 'first-order nesting treats the inner gradient as a constant');
  const secondOrder = values(await gradientArtifact(env, (await outer(2)).$gradient));
  assert.ok(norm(secondOrder.map((x, i) => x - firstOrder[i])) > 1e-6 * norm(firstOrder),
    'with { order: 2 } the inner gradient is differentiated: the outer gradient changes');
};

cases['grad-log-likelihood'] = async env => {
  const block = await env.produce('Write the instructions for answering tickets.');
  // A soft instruction block: the function's body is the native state itself.
  const answer = b => softFunction({ type: '(ticket: string) => Neuralese<string>', body: b, name: 'answer' });
  let recorded, callsBefore;
  // With `switchPolicy` the turns are recorded under the default policy and replayed under `alt`: forced choices give
  // the same loss and gradient, and the replay makes no generation request.
  const objective = switchPolicy => async b => {
    recorded = await env.learning.trajectory(() => env.runtime.run(() => answer(b)('refund')));
    if (switchPolicy) { await env.command('policy alt'); callsBefore = await env.command('calls'); }
    return env.learning.objectives.logLikelihood(recorded);
  };
  const plain = await env.learning.valueAndGrad(objective(false), block);
  const turn = recorded.turns.find(item => item.native?.kind === 'generation');
  const samples = turn.native.sampling.samples.map(sample => sample.token ?? sample.id ?? sample[0]);
  assert.ok(samples.length >= 4, `the turn sampled a producer span and its stop: ${JSON.stringify(turn.native.sampling.samples)}`);
  assert.equal(turn.native.sampling.writes?.length, 1, 'the turn wrote one native span');
  assert.ok(turn.native.graph.read_occurrences?.some(read => read.id === block.$neuralese.id) ||
    turn.native.graph.read_bindings?.[block.$neuralese.id] !== undefined, 'the turn read the soft instruction block');
  let replayed, callsAfter;
  try {
    replayed = await env.learning.valueAndGrad(objective(true), block);
    callsAfter = await env.command('calls');
  } finally { await env.command('policy default'); }
  assert.equal(callsAfter, callsBefore, 'replay made no generation request');
  assert.ok(Number.isFinite(+plain.loss) && +plain.loss > 0);
  assert.ok(Math.abs(+replayed.loss - +plain.loss) <= 1e-5 * Math.abs(+plain.loss), `replay forced the recorded choices: ${+plain.loss} vs ${+replayed.loss}`);
  const a = values(await gradientArtifact(env, plain.grad.$gradient)), b = values(await gradientArtifact(env, replayed.grad.$gradient));
  assert.ok(norm(a) > 0, 'a gradient at the block');
  close(a, b, 'the gradient under forced choices');
};

cases['graph-replay-fixed-choices'] = async env => {
  const skill = await env.produce('Write a skill for notes.');
  const notify = defineNatlang('---\nreturns: Neuralese<string>\n---\nNotify the team, then write the note.\n', { name: 'notify' });
  const consume = defineNatlang('---\nargs:\n  memo: Neuralese<string>\nreturns: string\n---\nRead the memo.\n', { name: 'consume' });
  env.scripts.push([/Notify the team/, 'const receipt = await mail.send("note ready");\nreceipt']);
  let trajectory;
  let callsBefore;
  const objective = switchPolicy => async s => {
    trajectory = await env.learning.trajectory(async () => {
      const memo = await env.runtime.run(() => notify.in(Context.ofCallable(notify).with({ skill: s }))());
      await env.runtime.run(() => consume(memo));
    });
    if (switchPolicy) { await env.command('policy alt'); callsBefore = await env.command('calls'); }
    return env.learning.objectives.logLikelihood(trajectory);
  };
  const effects = env.effects.length;
  const plain = await env.learning.valueAndGrad(objective(false), skill);
  assert.equal(env.effects.length, effects + 1, 'the recording ran the effect once');
  const writes = trajectory.turns.flatMap(turn => turn.native?.sampling?.writes ?? []);
  assert.ok(writes.length >= 1, 'the record holds a block write');
  const toolResult = trajectory.turns.flatMap(turn => turn.messages).find(message => message.role === 'tool' && /sent \d/.test(JSON.stringify(message.content)));
  assert.ok(toolResult, `a later turn of the record holds the effect's recorded result: ${JSON.stringify(trajectory.turns.map(turn => turn.messages.slice(-2)))}`);
  const consumer = trajectory.turns.find(turn => Object.keys(turn.native?.graph?.read_bindings ?? {}).some(id => writes.some(write => write.state_id === id)) ||
    turn.native?.graph?.read_occurrences?.some(read => writes.some(write => write.state_id === read.id) && read.source !== 'external'));
  assert.ok(consumer, 'the consumer reads the recorded write');
  for (const write of writes) {
    const state = await env.store.getNativeState(write.state_id);
    const tokens = state.tensors.token_ids;
    assert.ok(tokens, 'the written state keeps its span tokens');
    const length = tokens.shape.reduce((x, y) => x * y, 1);
    const event = trajectory.turns.flatMap(turn => Object.entries(turn.native?.graph?.written_nodes ?? {})).find(([id]) => id === write.state_id);
    assert.ok(event, 'the write is bound to its graph node');
    assert.equal(length, 1, 'the default policy writes one payload token');
  }
  let replayed, callsAfter;
  try {
    replayed = await env.learning.valueAndGrad(objective(true), skill);
    callsAfter = await env.command('calls');
  } finally { await env.command('policy default'); }
  assert.equal(env.effects.length, effects + 2, 'each recording runs the effect once; replay never calls the service');
  assert.equal(callsAfter, callsBefore, 'replay made no generation request');
  // Under `alt` a resampled producer would write two tokens and every stop would move: equal loss means the replay
  // used the recorded block length and tokens.
  assert.ok(Math.abs(+replayed.loss - +plain.loss) <= 1e-5 * Math.abs(+plain.loss), `replay used the recorded choices: ${+plain.loss} vs ${+replayed.loss}`);
  close(values(await gradientArtifact(env, plain.grad.$gradient)), values(await gradientArtifact(env, replayed.grad.$gradient)), 'the replayed gradient');
};

// Iteration (no server) --------------------------------------------------------------------------------------------
const recordingModel = answer => {
  const prompts = [];
  const model = ({ messages }) => { prompts.push(messages.map(message => String(message.content ?? '')).join('\n')); return { calls: [['return_result', { status: 'success', value: answer(prompts.length) }]] }; };
  return { model, prompts };
};

cases['iterate-ts-predicate-unbounded'] = async () => {
  let steps = 0;
  await assert.rejects(() => createNatlangRuntime().run(() => iterateOn(s => { steps++; return { done: s.n >= 3, n: s.n + 1 }; }, { done: false, n: 0 })
    .until(s => s.done)), error => error instanceof IterationLimitError && error.code === 'iteration-unbounded');
  assert.equal(steps, 0, 'rejected before the first step');
};

/** A compiled natlang program: `source` is main.ts, compiled like `natlang build` does. */
const compiled = source => {
  const result = compileVirtualProject({ files: { 'main.ts': source } }, natlang, { target: 'node' });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
  return result.require('main.ts');
};

cases['iterate-nl-predicate-unbounded'] = async () => {
  const program = compiled(`
    import { iterateOn, nl } from '@natlang/node';
    export const plan = () => iterateOn((s: { owners: number }) => ({ owners: s.owners + 1 }), { owners: 0 })
      .until(nl\`The plan names an owner for every task.\`);
  `);
  const { model, prompts } = recordingModel(count => count >= 3);
  const result = await createNatlangRuntime({ model }).run(() => program.plan());
  assert.deepEqual(result, { owners: 2 }, 'the iteration ran without a bound until the predicate held');
  assert.equal(prompts.length, 3);
  assert.match(prompts[0], /stopping condition of an iterative loop/);
  assert.match(prompts[0], /completed 0 steps\. This is the initial state/);
  assert.match(prompts[2], /completed 2 steps\./);
};

cases['iterate-judge-off-needs-bound'] = async () => {
  const program = compiled(`
    import { iterateOn, nl } from '@natlang/node';
    export const count = () => iterateOn((n: number) => n + 1, 0).checkProgress('off').until(nl\`The count is large enough.\`);
  `);
  const { model, prompts } = recordingModel(() => false);
  await assert.rejects(() => createNatlangRuntime({ model }).run(() => program.count()),
    error => error instanceof IterationLimitError && error.code === 'iteration-unbounded');
  assert.equal(prompts.length, 1, 'only the initial stopping check ran');
};

cases['iterate-ts-predicate-reviews-off'] = async () => {
  const { model, prompts } = recordingModel(() => ({ verdict: 'continue', reason: 'unexpected' }));
  const traces = [];
  const result = await createNatlangRuntime({ model, trace: trace => traces.push(trace) }).run(() =>
    iterateOn(s => ({ loss: s.loss / 2 }), { loss: 1 }).withLimit({ maxSteps: 50 }).checkProgress('off').until(s => s.loss < 0.1));
  assert.ok(result.loss < 0.1);
  assert.equal(prompts.length, 0, 'no model call at all');
  const events = traces.flatMap(trace => trace.events ?? []);
  assert.ok(!events.some(event => event.kind === 'iteration_step' && event.review), 'no progress-judge review in the trace');
  assert.ok(!events.some(event => event.kind === 'model_request'), 'no model request in the trace');
};

cases['learning-requires-service'] = async () => {
  const program = compiled(`
    import { grad } from 'natlang:learning';
    export const fit = async () => { await grad(async (s: unknown) => { throw new Error('unreachable'); }, null as never); return 1; };
  `);
  await assert.rejects(() => createNatlangRuntime().run(() => program.fit()),
    error => /learning-unavailable/.test(String(error?.message)) && /natlang:learning service/.test(String(error?.message)));
};
