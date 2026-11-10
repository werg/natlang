// Adopted model-facing variants (src/runtime/model-facing.ts) are on without any switch.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder, createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { session as open } from './support/natlang.mjs';

async function script(calls, runtimeOptions = {}) {
  const results = [];
  let step = 0;
  const model = async request => {
    if (step) results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ ...runtimeOptions, model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  return { results };
}

test('a successful eval lists the service calls it completed', async () => {
  const { results } = await script([['eval', { code: 'await shop.reserve("a"); await shop.reserve("b"); await shop.count(); 1' }]], {
    services: { shop: { reserve: () => true, count: () => 2 } } });
  assert.match(results[0], /Service calls completed by this eval \(not undone; do not repeat a write\): shop\.reserve x2, shop\.count\./);
});

test('an eval that calls no service has no receipt', async () => {
  const { results } = await script([['eval', { code: 'const x = 1; x' }]]);
  assert.doesNotMatch(results[0], /Service calls completed/);
});

test('read_file says how the file is bound in eval', async () => {
  const folder = Folder.fromFiles({ 'tickets/a.md': 'Status: open\n' });
  const opened = open({ type: '() => string', subtype: 'directory-reducer', instructions: 'Read the ticket.' }, {});
  opened.lam.projectTransaction = await folder.beginTransaction(false);
  const read = await opened.session.applyAsync('read_file', { path: 'tickets/a.md' });
  assert.equal(read.kind, 'ok', read.text);
  assert.ok(read.text.endsWith('\n(in eval: const text = await folder.file("tickets/a.md").readText())'), read.text);
  assert.equal(read.value, 'Status: open\n');
  opened.lam.projectTransaction.abort();
});

test('the system prompt keeps the shipped inline nl example', () => {
  assert.match(TOOLS_PROMPT, /Decide whether item meets policy\./);
});
