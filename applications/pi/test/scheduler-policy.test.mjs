/**
 * Wiring of the natural-language scheduler policy and admission (scripted executor: wiring evidence, not model
 * quality). pi-durable's Harness runs with its own built-in tasks and the faux provider; every scheduling decision and
 * every admission goes through scheduler/*.nl and admit.nl, whose scripted replies are the eval code an executor would
 * write: the decisions come from pi's rules restated over facts (`crispSchedulerPolicy`, exposed as a test service), the
 * cleanup and admission steps are written out as their instructions say.
 *
 * Run: node --test applications/pi/test/scheduler-policy.test.mjs (builds the app into .natlang/test-build first).
 */
import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build');
let m;

before(async () => {
  const natlang = await import(pathToFileURL(join(hostDist, 'index.js')).href);
  const runtime = pathToFileURL(join(hostDist, 'index.js'));
  const result = natlang.buildProject({ project: root, outDir, runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: join(hostDist, 'index.d.ts'), specifiers: ['@natlang/node'] } });
  if (!result.ok) throw new Error(natlang.formatDiagnostics(result.diagnostics));
  const load = path => import(pathToFileURL(join(outDir, path)).href);
  const { scriptedModel } = await import(pathToFileURL(join(hostDist, '../test/support/natlang.mjs')).href);
  m = { natlang, scriptedModel, durable: await load('vendor/durable/src/index.js'),
    policy: await load('vendor/durable/src/harness/policy.js'), policies: await load('host/policies.js'),
    live: await load('vendor/durable/src/harness/live.js'), memory: await load('vendor/durable/src/storage/memory.js'),
    ai: await import('@earendil-works/pi-ai'),
    chord: await import('@earendil-works/chord/context') };
});

/** Eval code per function, recognized by its instructions. */
const CODE = [
  ['Return the operations that clean up after task', `
    const live = await scheduler.live(task.conversationId);
    if (task.kind === 'pi.tool') { const slot = (live.tools ?? []).find(s => s.taskId === task.id); return slot ? [{ op: 'slot', callId: slot.callId, status: 'done' }] : []; }
    if (task.kind === 'pi.compaction') return [{ op: 'compactionStatus', remove: true }];
    if (task.kind === 'pi.generation' && live.run && live.run.taskId === task.id) return [{ op: 'convertPartial' }, { op: 'endRun', settlement:
      outcome.status === 'faulted' ? { status: 'unanswered', reason: 'faulted', detail: outcome.message } : { status: 'unanswered', reason: outcome.reason } }];
    return [];`],
  ['Classify every task of facts.tasks', `
    const d = oracle.pass(facts);
    for (const o of d.orphan) { const t = facts.tasks.find(t => t.id === o.id);
      o.cleanup = await cleanup({ id: t.id, kind: t.kind, conversationId: t.conversationId }, { status: 'orphaned', reason: o.reason }); }
    return d;`],
  ['Decide what happens to the invocation from facts', `
    const d = oracle.step(facts);
    if (d.kind === 'fault') d.cleanup = await cleanup({ id: facts.task.id, kind: facts.task.kind, conversationId: facts.task.conversationId }, { status: 'faulted', message: d.message });
    return d;`],
  ['Decide what the committed records imply', `
    const d = oracle.reconcile(facts);
    for (const f of d.finalize) { const t = facts.tasks.find(t => t.id === f.id);
      if (t.outcome === 'faulted' || t.outcome === 'orphaned') f.cleanup = await cleanup({ id: t.id, kind: t.kind, conversationId: t.conversationId }, { status: t.outcome, message: t.message, reason: t.reason }); }
    return d;`],
  ['Decide the abort of task facts.id', `
    const d = oracle.abortTask(facts);
    if (d.kind === 'orphan') d.cleanup = await cleanup({ id: facts.task.id, kind: facts.task.kind, conversationId: facts.task.conversationId }, { status: 'orphaned', reason: d.reason });
    return d;`],
  ['Decide the abort of conversation facts.conversation', 'return oracle.abortConversation(facts);'],
  ['Decide what a turn boundary places', `
    const reset = items.some(i => i.mode === 'write' && i.entry.head === 'self');
    const final = at === 'final' || reset;
    let start = activeStart; const writes = [];
    for (const i of items.filter(i => i.mode === 'write')) {
      const stale = typeof i.entry.head === 'number' && start !== null && i.entry.head < start;
      writes.push({ id: i.id, stale });
      if (!stale && i.entry.head !== undefined) start = i.entry.head === 'self' ? Number.MAX_SAFE_INTEGER : i.entry.head; }
    const pick = (mode, q) => { const ids = items.filter(i => i.mode === mode).map(i => i.id); return q === 'all' ? ids : ids.slice(0, 1); };
    const chosen = new Set([...pick('steer', steeringMode), ...(final ? pick('followUp', followUpMode) : [])]);
    return { writes, users: items.filter(i => chosen.has(i.id)).map(i => i.id), reset, final };`],
  ['Admit draft to the conversation of the admission service', `
    if (draft.requestId !== undefined) { const existing = await admission.byRequest(draft.requestId);
      if (existing) return existing.type !== draft.type ? { conflict: 'Request ' + draft.requestId + ' already identifies a submission of type ' + existing.type } : { id: existing.id }; }
    const state = await admission.state(); const busy = state.run !== null;
    if (busy && draft.type === 'input' && draft.whenBusy === 'reject') return { busy: true };
    const expect = { run: state.run, inbox: state.inbox.map(i => i.id), ...(draft.requestId !== undefined ? { requestAbsent: draft.requestId } : {}) };
    let ops;
    if (busy || state.inbox.length > 0) {
      ops = [{ op: 'queue', draft }];
      if (!busy) { const mode = draft.type === 'write' ? 'write' : draft.whenBusy === 'steer' ? 'steer' : 'followUp';
        const item = { id: 0, mode, ...(draft.type === 'write' ? { entry: draft.entry } : { content: draft.content }) };
        ops.push({ op: 'boundary', selection: await boundary([...state.inbox, item], state.steeringMode, state.followUpMode, 'final', state.activeStart) }); }
    } else if (draft.type === 'write') {
      const stale = typeof draft.entry.head === 'number' && state.activeStart !== null && draft.entry.head < state.activeStart;
      ops = [{ op: stale ? 'stale' : 'write', draft }];
    } else ops = [{ op: 'input', draft }];
    return { id: (await admission.commit(ops, expect)).id };`],
];

/** A tool that runs until its call is aborted. */
const untilAborted = async (_args, _api, context) => {
  await new Promise(resolve => { if (context.abortSignal?.aborted) resolve(); else context.abortSignal?.addEventListener('abort', resolve, { once: true }); });
  throw context.abortSignal?.reason ?? new Error('aborted');
};

function setup() {
  const calls = [];
  const scripted = m.scriptedModel(opening => {
    const found = CODE.find(([marker]) => opening.includes(marker));
    calls.push(found ? found[0].split(' ').slice(0, 4).join(' ') : 'unknown');
    if (process.env.DEBUG) console.error('call', calls.at(-1), found ? '' : opening.slice(0, 300));
    return found ? found[1] : null;
  });
  const natlang = m.natlang.createNatlangRuntime({ model: scripted.driver });
  const faux = m.ai.fauxProvider();
  const models = m.ai.createModels();
  models.setProvider(faux.provider);
  const registry = m.durable.createRegistry();
  const context = m.chord.BACKGROUND_CONTEXT;
  let harness;
  const host = { natlang, services: { oracle: m.policy.crispSchedulerPolicy }, now: () => Date.now(), onCall: event => { if (process.env.DEBUG && event.error) console.error('policy error', event); }, live: async (id, c) => (await harness.snapshot(m.live.LiveDoc, id, c)) ?? {} };
  const reports = [];
  const open = async () => {
    harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models, registry,
      schedulerPolicy: m.policies.schedulerPolicy(host, 'natural-language'), admission: m.policies.admissionPolicy(host, 'natural-language'),
      onReport: error => { reports.push(error); if (process.env.DEBUG) console.error('report', error); } }, context);
    const conversation = await harness.root(context, { agent: { model: { provider: 'faux', modelId: 'faux-1' } } });
    return { harness, conversation };
  };
  return { calls, faux, registry, context, open, reports };
}

test('an input is admitted, scheduled, stepped and settled through the natural-language policy', async () => {
  const s = setup();
  s.faux.setResponses([m.ai.fauxAssistantMessage('Hello there')]);
  const { harness, conversation } = await s.open();
  harness.resume();
  const submission = await conversation.submit({ type: 'input', content: 'hi' }, s.context);
  const settled = await submission.wait(s.context);
  assert.equal(settled.status, 'done');
  await harness.waitForIdle(s.context);
  const tasks = (await harness.commit(tx => tx.scanTasks({ conversationId: conversation.id }, 10), s.context)).items;
  assert.deepEqual(tasks.map(t => [t.kind, t.state.status, t.state.outcome?.status]), [['pi.generation', 'terminal', 'completed']]);
  for (const name of ['Admit draft to the', 'Classify every task of', 'Decide what happens to']) assert.ok(s.calls.includes(name), `${name} was called`);
  // A request ID admitted twice returns the first submission.
  const again = await conversation.submit({ type: 'input', content: 'hi', requestId: 'r1' }, s.context);
  assert.equal((await conversation.submit({ type: 'input', content: 'other', requestId: 'r1' }, s.context)).id, again.id);
  await again.wait(s.context);
  await harness.close(s.context);
});

test('a busy conversation queues a steer and rejects input that asks to be rejected', async () => {
  const s = setup();
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  s.registry.install(m.durable.defineExtension({ name: 'slow', tools: [m.durable.defineTool({ name: 'wait', description: 'waits',
    parameters: m.ai.Type.Object({}), execute: async () => { await gate; return { content: [{ type: 'text', text: 'waited' }] }; } })] }));
  s.faux.setResponses([m.ai.fauxAssistantMessage([m.ai.fauxToolCall('wait', {})], { stopReason: 'toolUse' }),
    m.ai.fauxAssistantMessage('after the steer'), m.ai.fauxAssistantMessage('done')]);
  const { harness, conversation } = await s.open();
  harness.resume();
  const first = await conversation.submit({ type: 'input', content: 'start' }, s.context);
  for (let i = 0; i < 200 && !(await harness.snapshot(m.live.LiveDoc, conversation.id, s.context))?.tools; i++) await new Promise(r => setTimeout(r, 10));
  await assert.rejects(conversation.submit({ type: 'input', content: 'no', whenBusy: 'reject' }, s.context), /busy/);
  const steer = await conversation.submit({ type: 'input', content: 'steer', whenBusy: 'steer' }, s.context);
  assert.equal((await steer.status(s.context)).status, 'queued');
  release();
  assert.equal((await first.wait(s.context)).status, 'done');
  assert.equal((await steer.wait(s.context)).status, 'done');
  await harness.close(s.context);
});

test('abortTask marks a running generation and its tool task; the abort handlers settle the run', async () => {
  const s = setup();
  s.registry.install(m.durable.defineExtension({ name: 'slow', tools: [m.durable.defineTool({ name: 'wait', description: 'waits',
    parameters: m.ai.Type.Object({}), execute: untilAborted })] }));
  s.faux.setResponses([m.ai.fauxAssistantMessage([m.ai.fauxToolCall('wait', {})], { stopReason: 'toolUse' })]);
  const { harness, conversation } = await s.open();
  harness.resume();
  const submission = await conversation.submit({ type: 'input', content: 'start' }, s.context);
  let live;
  for (let i = 0; i < 200 && !(live = await harness.snapshot(m.live.LiveDoc, conversation.id, s.context))?.tools?.[0]?.taskId; i++) await new Promise(r => setTimeout(r, 10));
  assert.equal(await harness.abortTask(live.run.taskId, s.context), 'marked');
  const settled = await submission.wait(s.context);
  assert.deepEqual([settled.status, settled.reason], ['unanswered', 'aborted']);
  assert.ok(s.calls.includes('Decide the abort of'), 'abortTask was called');
  assert.ok(s.calls.includes('Decide what the committed'), 'reconcile was called');
  await harness.close(s.context);
});

test('a phase that throws faults its task, and a task no definition can take is orphaned on abort', async () => {
  const s = setup();
  const boom = m.durable.defineTask({ name: 'test.boom', version: 1, initial: () => ({ phase: 'go' }),
    phases: { go: async () => { throw new Error('boom'); } }, abort: async () => {} });
  const idle = m.durable.defineTask({ name: 'test.idle', version: 1, initial: () => ({ phase: 'go' }),
    phases: { go: async (_task, runtime, context) => runtime.commit(() => ({ status: 'waiting', on: [], policy: 'allSettled', checkpoint: { phase: 'go', n: 1 } }), context) },
    abort: async () => {} });
  const extension = m.durable.defineExtension({ name: 'tasks', tasks: [boom, idle] });
  s.registry.install(extension);
  const { harness, conversation } = await s.open();
  harness.resume();
  const id = await conversation.commit(tx => tx.createTask(boom, {}, { ownership: { kind: 'conversation' } }), s.context);
  const faulted = await harness.waitForTask(id, s.context);
  assert.deepEqual(faulted.state.outcome, { status: 'faulted', error: { message: 'boom' } });
  // A pending task whose definition is gone is orphaned at once when aborted.
  s.registry.uninstall(extension);
  const orphanId = await conversation.commit(tx => tx.createTask(idle, {}, { ownership: { kind: 'conversation' } }), s.context);
  assert.equal(await harness.abortTask(orphanId, s.context), 'marked');
  const orphaned = await harness.waitForTask(orphanId, s.context);
  assert.deepEqual(orphaned.state.outcome, { status: 'orphaned', reason: 'missing_task' });
  await harness.close(s.context);
});

test('Conversation.abort withdraws queued inputs and marks the scope', async () => {
  const s = setup();
  s.registry.install(m.durable.defineExtension({ name: 'slow', tools: [m.durable.defineTool({ name: 'wait', description: 'waits',
    parameters: m.ai.Type.Object({}), execute: untilAborted })] }));
  s.faux.setResponses([m.ai.fauxAssistantMessage([m.ai.fauxToolCall('wait', {})], { stopReason: 'toolUse' })]);
  const { harness, conversation } = await s.open();
  harness.resume();
  const first = await conversation.submit({ type: 'input', content: 'start' }, s.context);
  for (let i = 0; i < 200 && !(await harness.snapshot(m.live.LiveDoc, conversation.id, s.context))?.tools?.[0]?.taskId; i++) await new Promise(r => setTimeout(r, 10));
  const queued = await conversation.submit({ type: 'input', content: 'later' }, s.context);
  await conversation.abort(s.context);
  assert.deepEqual([(await queued.status(s.context)).status, (await queued.status(s.context)).reason], ['unanswered', 'aborted']);
  assert.equal((await first.wait(s.context)).status, 'unanswered');
  assert.ok(s.calls.includes('Decide the abort of'), 'abortConversation was called');
  await harness.close(s.context);
});
