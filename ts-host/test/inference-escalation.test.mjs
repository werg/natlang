import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const report = { reason: 'Cannot complete the analysis', evidence: 'Observed the recorded value',
  attempted_work: 'Stored the value once', remaining_subproblem: 'Return the stored value plus one' };
const call = (name, args) => ({ calls: [[name, args]], completion_tokens: 1 });
const definition = body => loadVirtualNatlang({ 'run.nl': `---\nargs: {value: number}\nreturns: number\n---\n${body}\n` }, 'run.nl');

test('handoff continues the same scope and effect exactly once with actor and report in the graph', async () => {
  let writes = 0, n = 0;
  const traces = [], studentRequests = [], teacherRequests = [];
  const student = async request => {
    studentRequests.push(request);
    return ++n === 1 ? call('eval', { code: 'const observed = ledger.record(value);' }) : call('escalate', report);
  };
  const teacher = async request => {
    teacherRequests.push(request);
    return call('eval', { code: 'return observed + 1', finish: true });
  };
  const runtime = createNatlangRuntime({ calls: false, model: { driver: student, id: 'student-v1' },
    models: { expert: { driver: teacher, id: 'expert-v2' } }, escalation: { models: ['expert'] },
    services: { ledger: { record(value) { writes++; return value; } } },
    serviceDeclarations: { ledger: 'export function record(value: number): number;' }, trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => definition('Record the input once and return it plus one.')(41)), 42);
  assert.equal(writes, 1);
  assert.ok(studentRequests[0].tools.some(tool => tool.function.name === 'escalate'));
  assert.ok(!teacherRequests[0].tools.some(tool => tool.function.name === 'escalate'));
  assert.deepEqual(teacherRequests[0].tools, studentRequests[0].tools.filter(tool => tool.function.name !== 'escalate'));
  assert.ok(JSON.stringify(teacherRequests[0].messages).includes('ledger.record(value)'));
  const events = traces[0].events, handoff = events.find(e => e.kind === 'model_escalation');
  assert.deepEqual(handoff.report, report);
  assert.deepEqual(events.filter(e => e.kind === 'model_turn').map(e => e.actor.id), ['student-v1', 'student-v1', 'expert-v2']);
});

test('the graph request budget is shared between the first actor and its fallback', async () => {
  let calls = 0;
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelRequests: 1 },
    model: async () => { calls++; return call('escalate', report); },
    models: { expert: async () => { calls++; return call('return_result', { status: 'success', value: 42 }); } },
    escalation: { models: ['expert'] } });
  await assert.rejects(runtime.run(() => definition('Return the input.')(42)), /graph model request or token budget exhausted/);
  assert.equal(calls, 1);
});

test('configured fallback names and cycles fail closed', async () => {
  const student = async () => call('return_result', { status: 'success', value: 42 });
  for (const escalation of [{ models: ['missing'] }, { models: ['expert', 'expert'] }]) {
    const runtime = createNatlangRuntime({ calls: false, model: student, models: { expert: student }, escalation });
    await assert.rejects(runtime.run(() => definition('Return the input.')(42)), /not explicitly configured|cycle/);
  }
});

test('no configured fallback exposes no handoff tool', async () => {
  let names;
  const runtime = createNatlangRuntime({ calls: false, model: async request => {
    names = request.tools.map(tool => tool.function.name);
    return call('return_result', { status: 'success', value: 42 });
  } });
  assert.equal(await runtime.run(() => definition('Return the input.')(42)), 42);
  assert.ok(!names.includes('escalate'));
});

test('a child handoff returns its exact result to the parent and keeps ancestry', async () => {
  const traces = [], teachers = [];
  const fn = loadVirtualNatlang({
    'run.nl': '---\nargs: {value: number}\nreturns: number\n---\nReturn the child result.\n',
    'run/child.nl': '---\nargs: {value: number}\nreturns: number\n---\nReturn value plus one.\n',
  }, 'run.nl');
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelRequests: 3, maxModelTokens: 3 },
    model: { id: 'student', driver: async request => String(request.messages[1].content).includes('child(') ?
      call('escalate', report) : call('eval', { code: 'return await child(value)', finish: true }) },
    models: { expert: { id: 'expert', driver: async request => { teachers.push(request); return call('eval', { code: 'return value + 1', finish: true }); } } },
    escalation: { models: ['expert'] }, trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => fn(41)), 42);
  assert.equal(teachers.length, 1);
  const child = traces.find(t => t.parentCallId !== null), parent = traces.find(t => t.parentCallId === null);
  assert.equal(child.parentCallId, parent.callId);
  assert.ok(child.events.some(e => e.kind === 'model_escalation' && e.to === 'expert'));
});

test('foreign native values use their source text with explicit provenance', async () => {
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { neuraleseRef } = await import('../dist/native/neuralese.js');
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { session: open } = await import('./support/natlang.mjs');
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const neuralese = { store, dialect: 'native-state/fixture' };
  const session = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Use the memo.',
    args: { memo: neuraleseRef('Neuralese<string>', artifact.id) } }, { neuralese }).session;
  const student = Object.assign(async () => call('escalate', report), { neuralese: true });
  let sent;
  const expert = async request => { sent = request; return call('return_result', { status: 'success', value: 'done' }); };
  const agent = new NativeToolAgent(student, { neuralese, maxTurns: 3, actor: { id: 'student', revision: null },
    fallbacks: [{ name: 'expert', id: 'expert', driver: expert }] });
  await agent.run(session);
  assert.equal(session.lam.return, 'done');
  assert.ok(JSON.stringify(sent.messages).includes(artifact.metadata.text));
  assert.ok(!JSON.stringify(sent.messages).includes(artifact.id));
  const literal = session.runtime.trace.events.find(e => e.kind === 'literal' && e.source_kind === 'cross-model-source-text');
  assert.equal(literal.block, artifact.id);
  assert.equal(literal.target_actor, 'expert');
});

test('completion token budget remains shared after a handoff', async () => {
  let expertCalls = 0;
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelTokens: 1 },
    model: async request => { assert.equal(request.max_tokens, 1); return call('escalate', report); },
    models: { expert: async () => { expertCalls++; return call('return_result', { status: 'success', value: 42 }); } },
    escalation: { models: ['expert'] } });
  await assert.rejects(runtime.run(() => definition('Return the input.')(42)), /graph model request or token budget exhausted/);
  assert.equal(expertCalls, 0);
});

test('handoff is an action and cannot bypass the action budget', async () => {
  let expertCalls = 0, requests = 0;
  const runtime = createNatlangRuntime({ calls: false, limits: { maxActions: 1 },
    model: async () => ++requests === 1 ? call('eval', { code: 'const checked = value;' }) : call('escalate', report),
    models: { expert: async () => { expertCalls++; return call('return_result', { status: 'success', value: 42 }); } },
    escalation: { models: ['expert'] } });
  await assert.rejects(runtime.run(() => definition('Return the input.')(42)), /action or tool-call budget exhausted/);
  assert.equal(expertCalls, 0);
});

test('concurrent token reservations wait for released capacity without overspending', async () => {
  const { ModelBudget } = await import('../dist/native/model-budget.js');
  const budget = new ModelBudget(2, 2);
  const first = await budget.reserve(null);
  let secondResolved = false;
  const pending = budget.reserve(null).then(value => { secondResolved = true; return value; });
  await Promise.resolve();
  assert.equal(secondResolved, false);
  first.finish(1);
  const second = await pending;
  assert.equal(second.limit, 1);
  second.finish(1);
  assert.equal(await budget.reserve(null), undefined);
});

test('native source without text fails before the fallback receives a request', async () => {
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { neuraleseRef } = await import('../dist/native/neuralese.js');
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { session: open } = await import('./support/natlang.mjs');
  const fixture = nativeStateFixture(); fixture.metadata.text = '';
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(fixture);
  const neuralese = { store, dialect: 'native-state/fixture' };
  const session = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Use the memo.',
    args: { memo: neuraleseRef('Neuralese<string>', artifact.id) } }, { neuralese }).session;
  let expertCalls = 0;
  const student = Object.assign(async () => call('escalate', report), { neuralese: true });
  const agent = new NativeToolAgent(student, { neuralese, maxTurns: 3,
    fallbacks: [{ name: 'expert', id: 'expert', driver: async () => { expertCalls++; return {}; } }] });
  assert.match(await agent.run(session), /exact source text is unavailable/);
  assert.equal(expertCalls, 0);
});

test('active recorder captures both actors and the linked handoff graph', async () => {
  const { ModelTurnRecorder, setRecorderSource, activeRecorder } = await import('../dist/native/model-recording.js');
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { session: open } = await import('./support/natlang.mjs');
  const rec = new ModelTurnRecorder();
  setRecorderSource(() => rec);
  try {
    const session = open({ type: '() => string', instructions: 'Return a result.' }).session;
    const student = Object.assign(async request => {
      activeRecorder().record({ messages: structuredClone(request.messages), tools: structuredClone(request.tools),
        reply: { role: 'assistant', content: null }, blocks: [], native: {
          schema: 'natlang.native-recorded-turn/1', kind: 'generation', id: 'native-turn:one', retry: 0,
          sampling: { prompt: [], writes: [] } } });
      return { ...call('escalate', report), raw_response: { x_natlang_native_recorded_turn: 'native-turn:one' } };
    }, { neuralese: true });
    const expert = async () => call('return_result', { status: 'success', value: 'done' });
    const agent = new NativeToolAgent(student, { actor: { id: 'student', revision: 'a' },
      fallbacks: [{ name: 'expert', id: 'expert', revision: 'b', driver: expert }] });
    await agent.run(session);
    assert.equal(session.lam.return, 'done');
    const native = rec.nativeTurns();
    assert.deepEqual(native[0].native.actor, { id: 'student', revision: 'a' });
    const fallback = rec.turns.find(turn => turn.actor?.id === 'expert');
    assert.equal(fallback.actor.revision, 'b');
    const node = rec.graphNodes.get(fallback.graph_turn);
    const handoff = [...rec.graphNodes.values()].find(item => item.operation === 'model-escalation');
    assert.ok(node.inputs.some(input => input.node === handoff.node));
    assert.deepEqual(handoff.report, report);
  } finally { setRecorderSource(() => undefined); }
});

test('fallback native output crosses back as source text with its producer identity', async () => {
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { session: open } = await import('./support/natlang.mjs');
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture('expert-native'));
  const session = open({ type: '() => string', instructions: 'Return the memo.' }).session;
  const student = async () => call('escalate', report);
  const expert = Object.assign(async () => ({ calls: [['return_result', { status: 'success',
    value: [{ type: 'native_state', id: artifact.id }] }]], completion_tokens: 1,
    raw_response: { native_state: { blocks: [{ id: artifact.id }] } } }),
    { neuralese: true, neuraleseRuntime: async () => ({ store, dialect: 'native-state/expert-native' }) });
  const agent = new NativeToolAgent(student, { maxTurns: 3, actor: { id: 'student', revision: null },
    fallbacks: [{ name: 'expert', id: 'expert', driver: expert }] });
  await agent.run(session);
  assert.equal(session.lam.return, artifact.metadata.text);
  const events = session.runtime.trace.events;
  const write = events.find(e => e.kind === 'block_write' && e.block === artifact.id);
  const literal = events.find(e => e.source_kind === 'cross-model-output-source-text');
  assert.equal(literal.source_actor, 'expert');
  assert.ok(literal.inputs.some(input => input.node === write.node));
});

test('native fallback writes retain the expert actor and graph provenance in the recorder', async () => {
  const { ModelTurnRecorder, setRecorderSource, activeRecorder } = await import('../dist/native/model-recording.js');
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { session: open } = await import('./support/natlang.mjs');
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture('expert'));
  const session = open({ type: '() => string', instructions: 'Return the memo.' }).session;
  const rec = new ModelTurnRecorder(); setRecorderSource(() => rec);
  try {
    const student = async () => call('escalate', report);
    const expert = Object.assign(async request => {
      activeRecorder().record({ messages: structuredClone(request.messages), tools: structuredClone(request.tools),
        reply: { role: 'assistant', content: null }, blocks: [{ id: artifact.id }], native: {
          schema: 'natlang.native-recorded-turn/1', kind: 'generation', id: 'native-turn:expert', retry: 0,
          sampling: { prompt: [], writes: [{ state_id: artifact.id }] } } });
      return { calls: [['return_result', { status: 'success', value: [{ type: 'native_state', id: artifact.id }] }]],
        completion_tokens: 1, raw_response: { native_state: { blocks: [{ id: artifact.id }] },
          x_natlang_native_recorded_turn: 'native-turn:expert' } };
    }, { neuralese: true, neuraleseRuntime: async () => ({ store, dialect: 'native-state/expert' }) });
    await new NativeToolAgent(student, { actor: { id: 'student', revision: null },
      fallbacks: [{ name: 'expert', id: 'expert', revision: 'v2', driver: expert }] }).run(session);
    assert.equal(session.lam.return, artifact.metadata.text);
    const turn = rec.nativeTurns()[0];
    assert.deepEqual(turn.native.actor, { id: 'expert', revision: 'v2' });
    assert.ok(turn.native.graph.written_nodes[artifact.id]);
    assert.ok([...rec.graphNodes.values()].some(node => node.source_kind === 'cross-model-output-source-text'));
  } finally { setRecorderSource(() => undefined); }
});

test('foreign native soft results are re-encoded by the owning runtime', async () => {
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { createNativeStateFixturePort } = await import('./fixtures/native-state-port.mjs');
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { session: open } = await import('./support/natlang.mjs');
  const sourceStore = new MemoryNeuraleseStore(), artifact = await sourceStore.putNativeState(nativeStateFixture('expert'));
  const store = new MemoryNeuraleseStore(), neuralese = { store, dialect: 'native-state/fixture', port: createNativeStateFixturePort(store) };
  const session = open({ type: '() => Neuralese<string>', instructions: 'Return the memo.' }, { neuralese }).session;
  const expert = Object.assign(async () => ({ calls: [['return_result', { status: 'success',
    value: [{ type: 'native_state', id: artifact.id }] }]], completion_tokens: 1,
    raw_response: { native_state: { blocks: [{ id: artifact.id }] } } }),
    { neuralese: true, neuraleseRuntime: async () => ({ store: sourceStore, dialect: 'native-state/expert' }) });
  await new NativeToolAgent(async () => call('escalate', report), { neuralese, maxTurns: 3,
    actor: { id: 'student', revision: null }, fallbacks: [{ name: 'expert', id: 'expert', driver: expert }] }).run(session);
  const result = session.lam.return.$neuralese;
  assert.notEqual(result.id, artifact.id);
  const rewritten = await store.getNativeState(result.id);
  assert.equal(rewritten.metadata.text, artifact.metadata.text);
  assert.equal(rewritten.metadata.architecture, 'fixture');
  const events = session.runtime.trace.events;
  const literal = events.find(event => event.source_kind === 'cross-model-output-source-text');
  const write = events.find(event => event.kind === 'block_write' && event.block === result.id);
  assert.ok(write.inputs.some(input => input.node === literal.node));
});

test('model-specific adapters stop the handoff before the fallback request', async () => {
  const { setNativeAdapterSource } = await import('../dist/native/model-recording.js');
  const { NativeToolAgent } = await import('../dist/native/agent.js');
  const { session: open } = await import('./support/natlang.mjs');
  const session = open({ type: '() => string', instructions: 'Return a result.' }).session;
  let calls = 0;
  setNativeAdapterSource(() => [{ id: 'actor-bound-adapter', scale: 1 }]);
  try {
    const agent = new NativeToolAgent(async () => call('escalate', report), {
      fallbacks: [{ name: 'expert', id: 'expert', driver: async () => { calls++; return {}; } }] });
    assert.match(await agent.run(session), /model-specific adapters/);
    assert.equal(calls, 0);
  } finally { setNativeAdapterSource(() => undefined); }
});

test('a provider exceeding its reserved tokens cannot execute its proposed action', async () => {
  let effects = 0;
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelTokens: 1 },
    model: async () => ({ calls: [['eval', { code: 'ledger.record(value); return value;', finish: true }]], completion_tokens: 2 }),
    services: { ledger: { record() { effects++; } } }, serviceDeclarations: { ledger: 'export function record(value: number): void;' } });
  await assert.rejects(runtime.run(() => definition('Record and return the input.')(42)), /token budget exceeded/);
  assert.equal(effects, 0);
});

test('four identical failed actions hand off the live scope without repeating effects or resetting requests', async () => {
  let writes = 0, studentCalls = 0, expertCalls = 0;
  const traces = [];
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelRequests: 6 },
    model: { id: 'student', driver: async () => ++studentCalls === 1 ?
      call('eval', { code: 'const observed = ledger.record(value)' }) : call('eval', { code: 'missingName()' }) },
    models: { expert: { id: 'expert', driver: async () => { expertCalls++; return call('eval', { code: 'return observed + 1', finish: true }); } } },
    escalation: { models: ['expert'] }, services: { ledger: { record(value) { writes++; return value; } } },
    serviceDeclarations: { ledger: 'export function record(value: number): number;' }, trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => definition('Record once and return the input plus one.')(41)), 42);
  assert.equal(writes, 1); assert.equal(studentCalls, 5); assert.equal(expertCalls, 1);
  const handoff = traces[0].events.find(event => event.kind === 'model_escalation');
  assert.match(handoff.report.evidence, /repeated_failed_action/);
});

test('forced native text writers consume the same graph request budget', async () => {
  const { MemoryNeuraleseStore } = await import('../dist/native/neuralese-store.js');
  const { createNativeStateFixturePort } = await import('./fixtures/native-state-port.mjs');
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {}\nreturns: Neuralese<string>\n---\nReturn a memo.\n' }, 'run.nl');
  let writes = 0;
  const store = new MemoryNeuraleseStore(), fixture = createNativeStateFixturePort(store);
  const port = { dialect: fixture.dialect, forcedText: true, async write(...args) { writes++; return fixture.write(...args); } };
  const model = async () => call('return_result', { status: 'success', value: 'memo' });
  const blocked = createNatlangRuntime({ calls: false, model, limits: { maxModelRequests: 1 }, neuralese: { store, port } });
  await assert.rejects(blocked.run(() => fn()), /graph model request budget exhausted|graph model request or token budget exhausted/);
  assert.equal(writes, 0);
  const allowed = createNatlangRuntime({ calls: false, model, limits: { maxModelRequests: 2 }, neuralese: { store, port } });
  await allowed.run(() => fn());
  assert.equal(writes, 1);
});

test('no-progress handoff does not restart an exhausted graph budget', async () => {
  let studentCalls = 0, expertCalls = 0;
  const traces = [];
  const runtime = createNatlangRuntime({ calls: false, limits: { maxModelRequests: 4 },
    model: async () => { studentCalls++; return call('eval', { code: 'missingName()' }); },
    models: { expert: async () => { expertCalls++; return call('return_result', { status: 'success', value: 42 }); } },
    escalation: { models: ['expert'] }, trace: trace => traces.push(trace) });
  await assert.rejects(runtime.run(() => definition('Return the input.')(42)), /graph model request or token budget exhausted/);
  assert.equal(studentCalls, 4); assert.equal(expertCalls, 0);
  assert.ok(!traces[0].events.some(event => event.kind === 'model_escalation'));
});
