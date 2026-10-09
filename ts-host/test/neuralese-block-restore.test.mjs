/**
 * Blocks outlive server restarts (plans/neuralese/DECISIONS.md, "representation chosen by use"): the runtime's store is
 * the archive the Neuralese driver restores blocks from when a server answers `neuralese-unknown-block`, and requests
 * carry the session's owner. Against an in-process fake server that can forget its blocks.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { registerLocalEndpoint } from '../dist/model/chat-completion.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { decodeBlockBody, isUnknownBlockError, neuraleseServerModelTurn, OWNER_HEADER, restoreBlocks,
  withRestoredBlocks } from '../dist/model/neuralese-server.js';

const DIALECT = 'nd:natlang@1';
const block = value => ({ dialect: DIALECT, length: 1, width: 2, dtype: 'f32', data: new Uint8Array(new Float32Array([value, value]).buffer) });
const json = (status, value) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } });
const unknown = id => json(400, { error: { code: 'neuralese-unknown-block', message: `${id} is not in this server's store; PUT it first` } });

/** A fake Neuralese server: block endpoints and a decide endpoint that needs every block its messages name. */
function fakeServer(base) {
  const blocks = new Map(), log = [];
  const stop = registerLocalEndpoint(base, async (url, init) => {
    const path = new URL(url).pathname, method = init.method ?? 'GET';
    log.push({ method, path, owner: new Headers(init.headers).get(OWNER_HEADER) });
    const match = path.match(/^\/v1\/neuralese\/blocks\/(nz1_[a-z2-7]+)(\/meta)?$/);
    if (match && method === 'PUT') {
      const { meta, data } = decodeBlockBody(new Uint8Array(init.body));
      blocks.set(match[1], { meta, data });
      return json(201, meta);
    }
    if (match) return blocks.has(match[1]) ? json(200, blocks.get(match[1]).meta) : json(404, { error: { code: 'neuralese-unknown-block' } });
    if (path === '/v1/neuralese/decide') {
      const body = JSON.parse(init.body);
      for (const message of body.messages) for (const part of message.content) if (part.type === 'neuralese' && !blocks.has(part.id))
        return unknown(part.id);
      return json(200, { log_probs: body.options.map(() => -1), tokens: body.options.map(() => 1) });
    }
    return json(404, { error: { code: 'not-found' } });
  });
  return { blocks, log, stop, restart: () => blocks.clear() };
}

test('unknown-block errors are recognised in each form the clients raise them', () => {
  assert.ok(isUnknownBlockError(Object.assign(new Error('model HTTP 400: …'), { providerCode: 'neuralese-unknown-block' })));
  assert.ok(isUnknownBlockError(new Error('neuralese decide HTTP 400: {"error":{"code":"neuralese-unknown-block"}}')));
  assert.ok(isUnknownBlockError({ code: 'neuralese-unknown-block', message: 'x' }));
  assert.ok(!isUnknownBlockError(new Error('model HTTP 500: internal')));
});

test('restoreBlocks uploads what the server lacks from the archive and reports what neither has', async () => {
  const archive = new MemoryNeuraleseStore(), remote = new MemoryNeuraleseStore();
  const a = await archive.put(block(1)), b = await remote.put(block(2));
  assert.deepEqual(await restoreBlocks(remote, archive, [a.id, b.id, 'nz1_missing']), ['nz1_missing']);
  assert.ok(await remote.has(a.id));
});

test('withRestoredBlocks re-uploads and retries once when the server lost a block it was known to have', async () => {
  const archive = new MemoryNeuraleseStore(), remote = new MemoryNeuraleseStore();
  const a = await archive.put(block(3));
  const known = new Set();
  const run = async () => (await remote.has(a.id)) ? 'ok' : Promise.reject(new Error(`neuralese-unknown-block: ${a.id}`));
  assert.equal(await withRestoredBlocks(remote, archive, [a.id], run, known), 'ok');
  await remote.collect(new Set());  // the server restarted: the cache still says uploaded
  let calls = 0;
  assert.equal(await withRestoredBlocks(remote, archive, [a.id], () => { calls++; return run(); }, known), 'ok');
  assert.equal(calls, 2);
  await remote.collect(new Set());
  await assert.rejects(withRestoredBlocks(remote, new MemoryNeuraleseStore(), [a.id], run),
    /neuralese-unknown-block: .* is neither on the server nor in the runtime's store/);
  await assert.rejects(withRestoredBlocks(remote, archive, [], () => Promise.reject(new Error('model HTTP 500'))), /HTTP 500/);
});

test('the Neuralese driver restores blocks after a server restart and names its owner on every request', async () => {
  const base = 'http://neuralese-restore.local';
  const server = fakeServer(base);
  try {
    const store = new MemoryNeuraleseStore();
    const soft = await store.put(block(4));
    const driver = neuraleseServerModelTurn({ endpoint: base, model: 'fake', store, owner: 'session-1' });
    const ask = () => driver.decide({ messages: [{ role: 'user', content: [{ type: 'neuralese', id: soft.id }] }], options: ['yes', 'no'] });
    assert.deepEqual((await ask()).log_probs, [-1, -1]);
    server.restart();
    assert.deepEqual((await ask()).log_probs, [-1, -1]);
    const puts = server.log.filter(entry => entry.method === 'PUT');
    assert.equal(puts.length, 2);
    assert.ok(server.log.every(entry => entry.owner === 'session-1'));
    assert.equal(server.log.filter(entry => entry.path === '/v1/neuralese/decide').length, 3);  // one refused, retried
  } finally {
    server.stop();
  }
});
