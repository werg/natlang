// Run with Node 24: node --test test/   (pytest wrapper: tests/ops_dashboard/test_app.py)
// The page's modules against the installed Tendriller packages: domain and reducers, the agent program's duties (data service,
// feeds, edits), the inbox worker, and the browser-side helpers that can run without a browser.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const PY = process.env.NATLANG_OPS_PYTHON_TEST || 'python3';
const FAKE = join(HERE, 'fake_service.py');
const KEY = 'a'.repeat(64), OTHER = 'b'.repeat(64);

test('the domain binds every command to a declared topic and its reducer accepts writers only', async () => {
  const { definition, topics, imports, TOPICS, LOG_REDUCER_SOURCE } = await import('../pages/ops/domain.mjs');
  assert.equal(definition.id, 'natlang-ops');
  for (const name of ['proposeEdit', 'recordApplication', 'agentMessage', 'agentReceipt']) assert.ok(definition.commands[name], name);
  for (const name of ['edits', 'agentFeed']) assert.ok(definition.queries[name], name);
  const target = { kind: 'task', dataset: 'd', id: 't1' };
  for (const [name, input] of [['proposeEdit', { target, action: 'flag', text: 'why' }], ['recordApplication', { edit: 'x', status: 'applied', detail: 'ok' }]]) {
    const updates = definition.commands[name].prepare(input);
    assert.ok(updates.every(u => topics[u.topic]), `${name} writes a declared topic`);
  }
  assert.ok(Object.values(TOPICS).every(t => topics[t]) && topics['agent/message'] && topics['agent/receipt']);
  assert.ok(imports['app/ops-log'] === LOG_REDUCER_SOURCE);
  // held data is view-only; a reason is required for flag/exclude/correct
  assert.throws(() => definition.commands.proposeEdit.prepare({ target: { ...target, kind: 'held_task' }, action: 'flag', text: 'x' }), /HELD_DATA_IS_VIEW_ONLY/);
  assert.throws(() => definition.commands.proposeEdit.prepare({ target, action: 'exclude', text: '  ' }), /REASON_REQUIRED/);
  assert.throws(() => definition.commands.proposeEdit.prepare({ target, action: 'correct', text: 'r', correction: [] }), /CORRECTION_REQUIRED/);
  // the reducer retains a writer's input as an event on its configured topic and refuses a non-writer
  const reducer = (await import('data:text/javascript,' + encodeURIComponent(LOG_REDUCER_SOURCE))).default;
  const ctx = writer => ({ query: (_topic, q) => q.check === 'writer' && writer, config: { eventTopic: TOPICS.edit } });
  const ok = reducer.reduce({ count: 2 }, { author: KEY, payload: { a: 1 } }, ctx(true));
  assert.deepEqual([ok.ok, ok.state.count, ok.events[0].topic, ok.events[0].payload], [true, 3, TOPICS.edit, { a: 1 }]);
  assert.equal(reducer.reduce(null, { author: KEY, payload: {} }, ctx(false)).code, 'writer_required');
});

test('the agent program, inbox worker and page modules load against the installed packages', async () => {
  const agent = await import('../pages/ops/agent.mjs');
  assert.equal(typeof agent.openPage, 'function');
  assert.equal(typeof (await import('../pages/ops/inbox-worker.mjs')).main, 'function');
  assert.equal(typeof (await import('../shared/native-page.mjs')).openNativePage, 'function');
  assert.equal(typeof (await import('../shared/helpers/data-feed.mjs')).serveDataFeeds, 'function');
});

test('feed replies over 8000 characters are packed and unpacked losslessly', async () => {
  const { pack, unpack } = await import('../pages/ops/feeds.mjs');
  const small = { a: 1 }, big = { rows: Array.from({ length: 4000 }, (_, i) => ({ i, text: 'row ' + i })) };
  assert.equal(await pack(small), small);
  const packed = await pack(big);
  assert.deepEqual(Object.keys(packed), ['gz']);
  assert.ok(JSON.stringify(packed).length < JSON.stringify(big).length / 3);
  assert.deepEqual(await unpack(packed), big);
  assert.deepEqual(await unpack(small), small);
});

test('the data queries the agent program serves are exactly the data.* handlers of the Python service', async () => {
  const { DATA_QUERIES } = await import('../pages/ops/agent.mjs');
  const root = join(HERE, '..', '..');
  const handlers = new Set();
  for (const file of readdirSync(root).filter(f => /^data_.*\.py$/.test(f)))
    for (const m of readFileSync(join(root, file), 'utf8').matchAll(/@handler\('(data\.[a-z_]+)'\)/g)) handlers.add(m[1]);
  assert.deepEqual([...DATA_QUERIES].sort(), [...handlers].sort());
});

test('the data service child answers, errors and survives; feedSpecs serve the snapshot and data pulls', async () => {
  const { dataService, feedSpecs } = await import('../pages/ops/agent.mjs');
  const { unpack } = await import('../pages/ops/feeds.mjs');
  const stop = new AbortController(), logs = [];
  const rpc = dataService({ signal: stop.signal, log: o => logs.push(o), python: PY, script: FAKE, args: [] });
  try {
    assert.match((await rpc('ops.paths', null)).telemetry, /page-telemetry\.jsonl$/);
    await assert.rejects(rpc('nope', null), /nope/);
    const specs = feedSpecs(rpc);
    assert.deepEqual(Object.keys(specs).slice(0, 1), ['snapshot']);
    const snap = await unpack(await specs.snapshot.fetch());
    assert.equal(snap.v, 3);
    assert.equal(specs.snapshot.pollMs, 10_000);
    const item = await unpack(await specs['data.items'].fetch({ dataset: 'd' }));
    assert.deepEqual(item.input, { dataset: 'd' });
  } finally { stop.abort(); }
});

/** A session double with the parts reconcileEdits and the inbox pass use: query, command, operations, identity. */
function fakeSession(feedOrEdits) {
  const commands = [], known = new Map();
  return { commands, identity: { publicKey: KEY },
    async query(name) { return { value: typeof feedOrEdits === 'function' ? feedOrEdits(name) : feedOrEdits[name] }; },
    operations: { get: async id => known.get(id) ?? null },
    async command(name, input, { operationId }) {
      commands.push({ name, input, operationId });
      const op = { materialized: async () => ({ outcome: 'accepted' }) }; known.set(operationId, op); return op;
    } };
}

test('every retained edit is applied once and its application recorded with a bounded operation id', async () => {
  const { reconcileEdits, applyId } = await import('../pages/ops/agent.mjs');
  const long = 'native_' + 'x'.repeat(300);
  const edits = [{ operationId: 'e1', action: 'flag', target: { kind: 'task', dataset: 'd', id: 't' } }, { operationId: long, action: 'exclude', target: { kind: 'task', dataset: 'd', id: 'u' } }];
  const session = fakeSession({ edits: { edits, applications: [{ edit: 'e1', author: KEY }, { edit: long, author: OTHER }] } });
  const applied = [];
  const rpc = async (q, input) => { applied.push([q, input.operationId]); return { status: 'applied', detail: 'ok', files: 'f' }; };
  await reconcileEdits(session, rpc);
  // e1 has this agent's record already; the long one has only a foreign record, which does not count
  assert.deepEqual(applied, [['edits.apply', long]]);
  assert.equal(session.commands.length, 1);
  assert.equal(session.commands[0].name, 'recordApplication');
  assert.equal(session.commands[0].input.edit, long);
  assert.ok(session.commands[0].operationId.length < 60 && session.commands[0].operationId === applyId(edits[1]));
  await reconcileEdits(session, rpc);                      // the operation exists now: nothing is applied twice
  assert.equal(applied.length, 1);
  const failing = fakeSession({ edits: { edits: [edits[0]], applications: [] } });
  await reconcileEdits(failing, async () => { throw Error('boom'); });
  assert.equal(failing.commands[0].input.status, 'failed');
});

test('the page turns an ask into an agent message and the agent feed into asks, answers and receipts', async () => {
  const { toCommand, inboxOf } = await import('../pages/ops/ui/connect.mjs');
  const [name, input] = toCommand('ask', { kind: 'request_stop', subject: { kind: 'unit', id: 'u1', title: 'U' }, text: 'stop it' });
  assert.equal(name, 'agentMessage');
  assert.deepEqual([input.channel, input.to, input.text], ['chat', [], 'stop it']);
  assert.deepEqual(JSON.parse(input.data), { kind: 'request_stop', subject: { kind: 'unit', id: 'u1', title: 'U' } });
  assert.equal(toCommand('ask', { kind: 'ack_alert', subject: { kind: 'alert', id: 'a' }, text: 'ok' })[1].expectReply, false);
  assert.deepEqual(toCommand('proposeEdit', { x: 1 }), ['proposeEdit', { x: 1 }]);
  const op = 'ask-op-' + 'z'.repeat(200);                    // longer than the old 128-character reference
  const feed = { messages: [
    { author: OTHER, operationId: op, text: 'why?', atMs: 5, data: JSON.stringify({ kind: 'ask', subject: { kind: 'alert', id: 'a1' } }) },
    { author: OTHER, operationId: 'chat-1', text: 'hello', atMs: 6 },                                           // not a question
    { author: KEY, operationId: 'ans-1', text: 'because', atMs: 9, inReplyTo: { author: OTHER, operationId: op } }],
  receipts: { [`${OTHER}:${op}`]: [{ author: KEY, state: 'seen', atMs: 7 }] }, activity: [] };
  const { asks, answers } = inboxOf(feed, KEY);
  assert.equal(asks.length, 1);
  assert.deepEqual([asks[0].operationId, asks[0].kind, asks[0].subject.id, asks[0].receipts[0].state], [op, 'ask', 'a1', 'seen']);
  assert.deepEqual(answers.map(a => [a.ask, a.text]), [[op, 'because']]);
});

test('the inbox worker gives receipts, mirrors questions with answers to inbox.jsonl, and does not repeat itself', async () => {
  const { questionsOf, pass } = await import('../pages/ops/inbox-worker.mjs');
  const asked = (op, kind, id) => ({ author: OTHER, operationId: op, text: 'q ' + op, atMs: 1, data: JSON.stringify({ kind, subject: { kind: 'alert', id } }) });
  const feed = { messages: [asked('q1', 'ask', 'a'), asked('q2', 'ask', 'b'), asked('q3', 'ack_alert', 'c'),
    { author: KEY, operationId: 'ans-q2', text: 'answer two', atMs: 3, inReplyTo: { author: OTHER, operationId: 'q2' } }],
  receipts: { [`${OTHER}:q2`]: [{ author: KEY, state: 'seen' }] }, activity: [] };
  const session = fakeSession(name => name === 'agentFeed' ? feed : null);
  const dir = mkdtempSync(join(tmpdir(), 'ops-inbox-')), file = join(dir, 'inbox.jsonl');
  const qs = questionsOf(feed, KEY);
  assert.deepEqual(qs.map(q => [q.op, q.answers.length, q.receipts]), [['q1', 0, []], ['q2', 1, ['seen']], ['q3', 0, []]]);
  await pass(session, { file });
  // q1 gets `seen`, q2 (answered) gets `done`, q3 (an acknowledgement) gets `done` at once
  assert.deepEqual(session.commands.map(c => [c.input.messageOperationId, c.input.state]).sort(), [['q1', 'seen'], ['q2', 'done'], ['q3', 'done']]);
  const lines = readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse);
  assert.deepEqual(lines.map(l => [l.op, l.kind, l.answers.length]), [['q1', 'ask', 0], ['q2', 'ask', 1], ['q3', 'ack_alert', 0]]);
  assert.equal(lines[1].answers[0].text, 'answer two');
  assert.ok(!('receipts' in lines[0]));
  const before = session.commands.length;
  await pass(session, { file });                               // same state, same operation ids: nothing is submitted twice
  assert.equal(session.commands.length, before);
});

test('the page reads the snapshot schema version the exporter writes', async () => {
  const { SNAPSHOT_V } = await import('../pages/ops/ui/model.mjs');
  const m = readFileSync(join(HERE, '..', '..', 'export.py'), 'utf8').match(/'v': (\d+)/);
  assert.ok(m, 'export.py states its snapshot version');
  assert.equal(SNAPSHOT_V, Number(m[1]));
});
