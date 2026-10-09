/**
 * Stored view calls, forced per reader (host/views.ts; plans/neuralese/DECISIONS.md 2026-10-09, "representation chosen
 * by use"). pi-durable's Harness runs its own built-in tasks with the companion installed and the agent model behind the
 * natlang transport, served by a local fake Neuralese server (/v1/neuralese/info, /view, blocks, pin, collect, chat).
 * The agent's requests are then sent through the port's ai service (host/ai.ts `turn`), where forcing happens:
 * - a text reader reads the shaped text, and no view is written;
 * - a Neuralese reader reads one block per long output, memoized across turns and across a harness restart, written
 *   with the intent records.py's `intent()` gives and under the conversation's owner, pinned, collected, and restored
 *   from the runtime's store after the server forgot it;
 * - a view the server cannot write fails the turn loudly (retryable when the failure is transient).
 *
 * Run: node --test applications/pi/test/stored-views.test.mjs (builds the app into .natlang/test-build-views first).
 */
import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const repository = fileURLToPath(new URL('../../../', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build-views');
let m, server, endpoint;

/** The fake server's state: the blocks it holds, what it was sent, its next replies. */
const held = new Set();
const seen = [];
let replies = [];
/** The IDs /view answers with, in order (blocks the test put in the runtime's store), and a failure to answer with. */
let viewIds = [];
let viewFailure;

const chatReply = message => ({ id: 'r', choices: [{ finish_reason: message.tool_calls ? 'tool_calls' : 'stop', message }],
  usage: { prompt_tokens: 10, completion_tokens: 2 } });

before(async () => {
  const natlang = await import(pathToFileURL(join(hostDist, 'index.js')).href);
  const runtime = pathToFileURL(join(hostDist, 'index.js'));
  const result = natlang.buildProject({ project: root, outDir, runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: join(hostDist, 'index.d.ts'), specifiers: ['@natlang/node'] } });
  if (!result.ok) throw new Error(natlang.formatDiagnostics(result.diagnostics));
  const load = path => import(pathToFileURL(join(outDir, path)).href);
  const { scriptedModel } = await import(pathToFileURL(join(hostDist, '../test/support/natlang.mjs')).href);
  m = { natlang, scriptedModel, durable: await load('vendor/durable/src/index.js'), companion: await load('extensions/companion/index.js'),
    views: await load('host/views.js'), aiHost: await load('host/ai.js'), main: await load('main.js'),
    provider: await load('vendor/durable/src/harness/provider.js'), sqlite: await load('vendor/durable/src/storage/sqlite/node.js'),
    chord: await import('@earendil-works/chord/context') };
  server = createServer((request, response) => {
    const chunks = [];
    request.on('data', chunk => chunks.push(chunk));
    request.on('end', () => {
      const raw = Buffer.concat(chunks);
      const json = (request.headers['content-type'] ?? '').includes('json') && raw.length ? JSON.parse(raw.toString()) : undefined;
      const path = request.url;
      seen.push({ method: request.method, path, body: json, owner: request.headers['x-natlang-owner'] });
      const send = (status, body) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
      const block = path.match(/^\/v1\/neuralese\/blocks\/([^/]+)(?:\/(meta|pin|unpin))?$/);
      if (path === '/v1/neuralese/info') return send(200, { dialects: ['d1'], width: 8, dtype: 'f32', max_block_length: 64 });
      if (path === '/v1/neuralese/view') {
        if (viewFailure) return send(viewFailure.status, { error: { message: viewFailure.message } });
        const id = viewIds.shift();
        held.add(id);
        return send(200, { id, dialect: 'd1', length: 5, width: 8, dtype: 'f32' });
      }
      if (block && request.method === 'PUT') { held.add(block[1]); return send(200, { id: block[1], dialect: 'd1', length: 5, width: 8, dtype: 'f32' }); }
      if (block?.[2] === 'meta') return held.has(block[1]) ? send(200, { id: block[1], dialect: 'd1', length: 5, width: 8, dtype: 'f32' }) :
        send(404, { error: { code: 'neuralese-unknown-block', message: 'neuralese-unknown-block' } });
      if (block?.[2] === 'pin') return held.has(block[1]) ? send(200, {}) : send(404, { error: { message: `neuralese-unknown-block: ${block[1]}` } });
      if (block?.[2] === 'unpin') return send(200, {});
      if (path === '/v1/neuralese/collect') return send(200, { removed: [] });
      if (path === '/v1/chat/completions') {
        const missing = json.messages.flatMap(message => Array.isArray(message.content) ? message.content : [])
          .filter(part => part.type === 'neuralese' && !held.has(part.id)).map(part => part.id);
        if (missing.length) return send(400, { error: { code: 'neuralese-unknown-block', message: `neuralese-unknown-block: ${missing.join(', ')}` } });
        return send(200, chatReply(replies.shift() ?? { content: 'ok' }));
      }
      send(404, { error: { message: `not found: ${path}` } });
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  endpoint = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

const context = () => m.chord.BACKGROUND_CONTEXT;
const SHORT = 'a'.repeat(3_000);
const LONG = Array.from({ length: 1_500 }, (_, i) => `line ${i}`).join('\n');
const FIRST_TURN = { content: 'Listing both.', reasoning_content: 'I need both listings.', tool_calls: [
  { id: 'c1', type: 'function', function: { name: 'bash', arguments: '{"command":"short"}' } },
  { id: 'c2', type: 'function', function: { name: 'bash', arguments: '{"command":"long"}' } }] };

/** A stored block of `length` vectors of width 8 in dialect d1, and its ID. */
async function storedBlock(store, length, seed) {
  const data = new Float32Array(length * 8).map((_, i) => seed + i);
  return (await store.put({ dialect: 'd1', length, width: 8, dtype: 'f32', data: new Uint8Array(data.buffer) })).id;
}

/** A Harness on `storagePath` with the companion and a bash tool printing SHORT or LONG; the agent reads `reader`. */
async function open(storagePath, store, reader) {
  const scripted = m.scriptedModel(opening => opening.includes('You accompany a coding agent')
    ? `return { focus: 'x', facts: [], warnings: [], suggestions: [] };` : null);
  const natlang = m.natlang.createNatlangRuntime({ model: scripted.driver, neuralese: { store, dialect: 'd1' } });
  const { models, ref } = await m.main.agentModels(['--agent-endpoint', endpoint, '--agent-model', 'fake', '--agent-transport', 'natlang',
    '--agent-reader', reader], undefined, store);
  const registry = m.durable.createRegistry();
  registry.install({ name: 'tools', tools: [{ name: 'bash', description: 'Run', parameters: { type: 'object', properties: { command: { type: 'string' } } },
    execute: async args => ({ content: [{ type: 'text', text: args.command === 'long' ? LONG : SHORT }] }) }] });
  let harness;
  const reports = [];
  registry.install(m.companion.companion(natlang, { harness: () => harness, onReport: error => reports.push(String(error)) }));
  harness = await m.durable.Harness.open(await m.sqlite.openNodeSqliteStorage(storagePath), { models, registry,
    onReport: error => reports.push(String(error)) }, context());
  const conversation = await harness.root(context(), { agent: { model: ref } });
  harness.resume();
  /** The port's ai service for a generation task of the conversation (host/tasks.ts builds it so). */
  const runtime = { models, signal: undefined, conversationId: conversation.id,
    snapshot: (...args) => harness.snapshot(...args), commit: (change, at) => conversation.commit(tx => change(tx, undefined), at),
    context: (_id, at) => conversation.context(at), report: error => reports.push(String(error)) };
  const ai = m.aiHost.aiService(runtime, context(), undefined, {}, store, { collect: true });
  const owner = async () => (await harness.snapshot(m.provider.ProviderDoc, conversation.id, context())).sessionId;
  /** One request of the conversation's context through the port's ai service. */
  const turn = async () => ai.turn(ref, [...(await conversation.context(context())).messages], { thinkingLevel: 'off', sessionId: await owner() });
  return { harness, conversation, ai, turn, owner, reports };
}

const toolResults = async conversation => (await conversation.entries({ order: 'ascending' }, 50, undefined, context())).items
  .flatMap(entry => entry.model ?? []).filter(message => message.role === 'toolResult');
const chats = () => seen.filter(item => item.path === '/v1/chat/completions');
const views = () => seen.filter(item => item.path === '/v1/neuralese/view');
const toolMessages = request => request.body.messages.filter(message => message.role === 'tool');

test('a text reader reads the shaped text of a stored view call, and no view is written', async () => {
  seen.length = 0;
  replies = [FIRST_TURN, { content: 'done' }];
  const store = new m.natlang.MemoryNeuraleseStore();
  const session = await open(join(mkdtempSync(join(tmpdir(), 'pi-views-')), 'session.db'), store, 'text');
  assert.equal((await (await session.conversation.submit({ type: 'input', content: 'list' }, context())).wait(context())).status, 'done');
  const [short, long] = await toolResults(session.conversation);
  // Both outputs are stored calls: the short one shown whole, the long one shaped; the reference rides on the text part.
  assert.deepEqual(short.content, [{ type: 'text', text: SHORT, stored: { function: 'view', call: 'c1' } }]);
  assert.deepEqual(long.content[0].stored, { function: 'view', call: 'c2' });
  assert.match(long.content[0].text, /^line 0\n[\s\S]*recall\("c2"\) returns the full output[\s\S]*line 1499$/);
  assert.equal((await session.harness.snapshot(m.views.ToolOutputs, session.conversation.id, 'c2', context())).text, LONG);
  // No intent is recorded for a text reader, and the port's request carries the text.
  assert.equal(await session.harness.snapshot(m.views.ViewIntents, session.conversation.id, 'c1', context()), undefined);
  const before = chats().length;
  const answer = await session.turn();
  assert.equal(answer.stopReason, 'stop', answer.errorMessage);
  const [request] = chats().slice(before);
  assert.deepEqual(toolMessages(request).map(message => message.content), [SHORT, long.content[0].text]);
  assert.equal(views().length, 0);
  assert.deepEqual(session.reports, []);
  await session.harness.close(context());
});

test('a Neuralese reader reads one view block per long output, memoized across turns and a restart', async () => {
  seen.length = 0;
  held.clear();
  replies = [FIRST_TURN, { content: 'done' }];
  const store = new m.natlang.MemoryNeuraleseStore();
  const blocks = [await storedBlock(store, 5, 1), await storedBlock(store, 7, 2)];
  viewIds = [...blocks];
  viewFailure = undefined;
  const path = join(mkdtempSync(join(tmpdir(), 'pi-views-')), 'session.db');
  let session = await open(path, store, 'd1');
  assert.equal((await (await session.conversation.submit({ type: 'input', content: 'list' }, context())).wait(context())).status, 'done');
  const owner = await session.owner();

  // The intent of each call, as the harness bench's records.py intent() writes it.
  const intents = [];
  for (const call of ['c1', 'c2']) intents.push((await session.harness.snapshot(m.views.ViewIntents, session.conversation.id, call, context())).intent);
  const assistant = { content: [{ type: 'thinking', thinking: FIRST_TURN.reasoning_content }, { type: 'text', text: FIRST_TURN.content }] };
  const expected = JSON.parse(execFileSync('python3', ['-I', '-c', `import json, sys
sys.path.insert(0, ${JSON.stringify(join(repository, 'training/neuralese'))})
from natlang_neuralese.harness_bench.records import intent
assistant = json.load(sys.stdin)
print(json.dumps([intent(assistant, {'name': 'bash', 'arguments': {'command': c}}) for c in ('short', 'long')]))`],
  { input: JSON.stringify(assistant) }).toString());
  assert.deepEqual(intents, expected);
  assert.match(intents[1], /^The agent made this tool call and reads its output next:\nbash \{"command":"long"\}\nIts reasoning when it made the call:\nI need both listings\.\nListing both\.$/);

  // The request: each long output is its view block and the recall note; the views were written under the owner.
  const before = chats().length;
  const answer = await session.turn();
  assert.equal(answer.stopReason, 'stop', answer.errorMessage);
  const [request] = chats().slice(before);
  const written = Object.fromEntries(views().map(view => [view.body.value === LONG ? 'c2' : 'c1', view]));
  assert.equal(views().length, 2);
  assert.deepEqual(views().map(view => view.owner), [owner, owner]);
  assert.equal(written.c1.body.instructions, intents[0]);
  assert.equal(written.c2.body.instructions, intents[1]);
  const memo = {};
  for (const call of ['c1', 'c2']) memo[call] = (await session.harness.snapshot(m.views.ViewForcings, session.conversation.id, call, context())).forced['neuralese:d1'];
  assert.deepEqual(new Set([memo.c1.id, memo.c2.id]), new Set(blocks));
  assert.equal(memo.c1.length, blocks[0] === memo.c1.id ? 5 : 7);
  assert.equal(memo.c1.dialect, 'd1');
  assert.deepEqual(toolMessages(request).map(message => message.content), ['c1', 'c2'].map(call =>
    [{ type: 'neuralese', id: memo[call].id }, { type: 'text', text: `  // view of the output; recall("${call}") returns all of it` }]));
  assert.equal(request.owner, owner, 'the chat request names the conversation as the owner of its blocks');
  // Pinned under the owner, and collected with exactly the referenced blocks at the first request.
  assert.deepEqual(new Set(seen.filter(item => item.path.endsWith('/pin')).map(item => [item.path.split('/')[4], item.owner].join(' '))),
    new Set(blocks.map(id => `${id} ${owner}`)));
  const collects = seen.filter(item => item.path === '/v1/neuralese/collect');
  assert.equal(collects.length, 1);
  assert.equal(collects[0].owner, owner);
  assert.deepEqual(new Set(collects[0].body.referenced), new Set(blocks));
  // The stored message is unchanged: text for text readers, with the reference.
  assert.deepEqual((await toolResults(session.conversation))[0].content, [{ type: 'text', text: SHORT, stored: { function: 'view', call: 'c1' } }]);

  // A second request: the memo, no new view, no new collection (the context's head did not move).
  assert.equal((await session.turn()).stopReason, 'stop');
  assert.equal(views().length, 2);
  assert.equal(seen.filter(item => item.path === '/v1/neuralese/collect').length, 1);
  assert.deepEqual(session.reports, []);
  await session.harness.close(context());

  // A restart: the memo is durable.
  session = await open(path, store, 'd1');
  assert.equal(await session.owner(), owner);
  const reopened = chats().length;
  assert.equal((await session.turn()).stopReason, 'stop');
  assert.equal(views().length, 2);
  assert.deepEqual(toolMessages(chats()[reopened]).map(message => message.content[0].id), [memo.c1.id, memo.c2.id]);

  // The server restarts and forgets its blocks: the request restores them from the runtime's store and goes through.
  held.clear();
  const puts = () => seen.filter(item => item.method === 'PUT').map(item => item.path.split('/')[4]);
  assert.deepEqual(puts(), []);
  const answerAfter = await session.turn();
  assert.equal(answerAfter.stopReason, 'stop', answerAfter.errorMessage);
  assert.deepEqual(new Set(puts()), new Set(blocks));
  assert.equal(views().length, 2);
  assert.deepEqual(session.reports, []);
  await session.harness.close(context());
});

test('a view the server cannot write fails the turn loudly; a transient failure is retryable', async () => {
  seen.length = 0;
  held.clear();
  replies = [FIRST_TURN, { content: 'done' }];
  const store = new m.natlang.MemoryNeuraleseStore();
  viewFailure = { status: 503, message: 'overloaded' };
  const session = await open(join(mkdtempSync(join(tmpdir(), 'pi-views-')), 'session.db'), store, 'd1');
  assert.equal((await (await session.conversation.submit({ type: 'input', content: 'list' }, context())).wait(context())).status, 'done');
  // The hook's pre-force failed and was reported; the request forces again, and fails without being sent.
  for (let i = 0; i < 100 && !session.reports.some(report => report.includes('neuralese-view')); i++) await new Promise(done => setTimeout(done, 20));
  assert.ok(session.reports.some(report => report.includes('neuralese-view-unavailable')), session.reports.join('; '));
  const before = chats().length;
  const transient = await session.turn();
  assert.equal(transient.stopReason, 'error');
  assert.match(transient.errorMessage, /^neuralese-view-unavailable: the view of tool call c\d's output for the "d1" reader .* view failed \(503\)/);
  assert.deepEqual(session.ai.failure(transient), { overflow: false, retryable: true });
  viewFailure = { status: 400, message: 'value too long' };
  const permanent = await session.turn();
  assert.match(permanent.errorMessage, /^neuralese-view-failed: .* view failed \(400\)/);
  assert.deepEqual(session.ai.failure(permanent), { overflow: false, retryable: false });
  assert.equal(chats().length, before, 'no request reaches the model with text in place of a view');
  await session.harness.close(context());
});

test('a collection unpins the view blocks that left the context once the head moved', async () => {
  held.clear();
  const state = { pinned: { nz1_kept: 'c1', nz1_gone: 'c2' }, collectedHead: 3 };
  const docs = { conversationId: 1, read: { snapshot: async () => state },
    commit: async change => change({ doc: async () => state }) };
  const models = (await m.main.agentModels(['--agent-endpoint', endpoint, '--agent-model', 'fake', '--agent-transport', 'natlang',
    '--agent-reader', 'd1'])).models;
  const reader = { models, model: models.getModel('agent', 'fake'), owner: 'o1' };
  seen.length = 0;
  // Requests of this owner only: a forcing of an earlier test may still be settling.
  const mine = () => seen.filter(item => item.owner === 'o1');
  await m.views.collectViewBlocks(docs, reader, new Set(['nz1_kept']), 3, context());
  assert.deepEqual(mine(), [], 'the same head: nothing to collect');
  await m.views.collectViewBlocks(docs, reader, new Set(['nz1_kept', 'nz1_other']), 9, context());
  assert.deepEqual(mine().map(item => [item.method, item.path]), [['POST', '/v1/neuralese/blocks/nz1_gone/unpin'],
    ['POST', '/v1/neuralese/collect']]);
  assert.deepEqual(mine()[1].body.referenced, ['nz1_kept', 'nz1_other']);
  assert.deepEqual(state, { pinned: { nz1_kept: 'c1' }, collectedHead: 9 });
});
