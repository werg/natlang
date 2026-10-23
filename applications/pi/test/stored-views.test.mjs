/** Native-state custody for stored views: pin release and owner collection use only the current native archive routes. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build-native-views');

test('view collection unpins expired native states and collects the exact live native-state references', async () => {
  const natlang = await import(pathToFileURL(join(hostDist, 'index.js')).href);
  const runtime = pathToFileURL(join(hostDist, 'index.js'));
  const built = natlang.buildProject({ project: root, outDir, runtimeModule: { url: runtime.href,
    path: fileURLToPath(runtime), types: join(hostDist, 'index.d.ts'), specifiers: ['@natlang/node'] } });
  if (!built.ok) throw new Error(natlang.formatDiagnostics(built.diagnostics));
  const views = await import(pathToFileURL(join(outDir, 'host/views.js')).href);
  const calls = [];
  const server = createServer((request, response) => {
    const chunks = [];
    request.on('data', chunk => chunks.push(chunk));
    request.on('end', () => {
      const body = Buffer.concat(chunks).toString();
      calls.push({ method: request.method, path: request.url, body: body ? JSON.parse(body) : undefined,
        owner: request.headers['x-natlang-owner'] });
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ removed: [] }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const id = 'nzs1_' + 'a'.repeat(52), expired = 'nzs1_' + 'b'.repeat(52);
    const state = { pinned: { [expired]: 'call-old' }, collectedHead: 3 };
    const docs = { conversationId: 1, read: { snapshot: async () => structuredClone(state) },
      commit: async change => change({ doc: async () => state }) };
    const endpoint = `http://127.0.0.1:${server.address().port}`;
    const reader = { models: { getAuth: async () => undefined },
      model: { provider: 'agent', id: 'fixture', baseUrl: `${endpoint}/v1` }, owner: 'conversation-7' };
    await views.collectViewNativeStates(docs, reader, new Set([id]), 9, natlang.BACKGROUND_CONTEXT);
    assert.deepEqual(calls.map(({ method, path }) => [method, path]), [
      ['POST', `/v1/neuralese/native-states/${expired}/unpin`], ['POST', '/v1/neuralese/collect']]);
    assert.equal(calls[0].owner, 'conversation-7');
    assert.deepEqual(calls[1].body, { referenced: [id] });
    assert.deepEqual(state, { pinned: {}, collectedHead: 9 });
  } finally { await new Promise(resolve => server.close(resolve)); }
});
