// The service boundary: arguments arrive as host-realm plain data, and ONCE_EFFECTS methods act once per arguments.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';
import { TypeEnv, parseType, ONCE_EFFECTS } from '../dist/index.js';
import { coerce, Reject } from '../dist/native/values.js';
import { recordingServices } from '../dist/native/effects.js';

test('service arguments from the eval realm reach the service as host-realm plain data', () => {
  let seen;
  const services = recordingServices({ store: { put(value) { seen = value; } } }, () => {});
  const sandboxed = runInNewContext('({ items: [1, { tags: ["a"] }] })');
  assert.notEqual(Object.getPrototypeOf(sandboxed.items), Array.prototype);
  services.store.put(sandboxed);
  assert.equal(Object.getPrototypeOf(seen.items), Array.prototype);
  assert.equal(Object.getPrototypeOf(seen.items[1].tags), Array.prototype);
  assert.equal(Object.getPrototypeOf(seen), Object.prototype);
  assert.deepEqual(seen, { items: [1, { tags: ['a'] }] });
});

test('a once effect acts once per distinct arguments; failures and other methods may repeat', async () => {
  const calls = [];
  let fail = true;
  const target = {
    [ONCE_EFFECTS]: ['send', 'flaky'],
    async send(to, body) { calls.push(['send', to, body]); return { id: calls.length }; },
    async flaky() { calls.push(['flaky']); if (fail) { fail = false; throw new Error('down'); } return 'ok'; },
    read() { calls.push(['read']); return calls.length; },
  };
  const services = recordingServices({ mail: target }, () => {});
  const first = await services.mail.send('a', { text: 'hi' });
  assert.deepEqual(await services.mail.send('a', { text: 'hi' }), first);
  await services.mail.send('b', { text: 'hi' });
  await assert.rejects(services.mail.flaky(), /down/);
  assert.equal(await services.mail.flaky(), 'ok');
  assert.equal(await services.mail.flaky(), 'ok');
  services.mail.read(); services.mail.read();
  assert.deepEqual(calls.map(call => call[0]), ['send', 'send', 'flaky', 'flaky', 'read', 'read']);
  // Another wrapper over the same service object shares its once results (nested calls get their own wrappers).
  const again = recordingServices({ mail: target }, () => {});
  assert.deepEqual(await again.mail.send('a', { text: 'hi' }), first);
  assert.equal(calls.length, 6);
});

test('a union of records reports the wrong field of the variant the value matches', () => {
  const type = parseType('{ phase: "prepare"; attempt: number } | { phase: "request"; attempt: number; cutoff: number }');
  assert.throws(() => coerce({ phase: 'request', attempt: 1, cutoff: '7' }, type, new TypeEnv(), 'checkpoint'), error => {
    assert.ok(error instanceof Reject);
    assert.equal(error.diagnostics[0].path, 'checkpoint/cutoff');
    assert.match(error.message, /checkpoint\/cutoff: type-mismatch, expected number \(in .*\), got the string "7"; write the number itself/);
    return true;
  });
  // A value no variant comes close to still reports the union.
  assert.throws(() => coerce({ phase: 'retry' }, type, new TypeEnv(), 'checkpoint'), /^Error|checkpoint: type-mismatch, expected \{/);
});

test('a value eval computed keeps the fields its declared record does not list; a return_result literal is exact', async () => {
  const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');
  const run = async calls => {
    let step = 0;
    const model = async () => ({ calls: [calls[step++] ?? ['return_result', { status: 'failed', reason: 'no more calls' }]] });
    const provider = { reply() { return { role: 'assistant', text: 'hi', durationMs: 12 }; } };
    const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' }, services: { provider } });
    const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: "{ role: string, text: string }"\n---\nAsk provider.\n' }, 'root.nl');
    try { return await runtime.run(() => fn('Ada')); } catch (error) { return error; }
  };
  assert.deepEqual(await run([['eval', { code: 'const message = provider.reply(); return message;', finish: true }]]),
    { role: 'assistant', text: 'hi', durationMs: 12 });
  const literal = await run([['return_result', { status: 'success', value: { role: 'assistant', text: 'hi', extra: 1 } }]]);
  assert.ok(literal instanceof Error, 'a return_result literal with an unknown field is refused');
});

test('an eval local keeps the fields of a computed value its type does not list', async () => {
  const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');
  let step = 0;
  const calls = [['eval', { code: 'const result: { items: { type: "call"; id: string }[] } = view.view(); result.items.length' }],
    ['eval', { code: 'return result;', finish: true }]];
  const model = async () => ({ calls: [calls[step++] ?? ['return_result', { status: 'failed', reason: 'no more calls' }]] });
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  // The helper's declared return type lists fewer fields than the value carries, as a provider's message can.
  const fn = loadVirtualNatlang({
    'root.nl': '---\nargs: {}\nreturns: "{ items: { type: \\"call\\", id: string }[] }"\n---\nReturn the view.\n',
    'root/view.ts': 'export function view(): { items: { type: "call"; id: string }[] } {\n' +
      '  return { items: [{ type: "call", id: "c1", signature: "sig" }] } as never;\n}\n' }, 'root.nl');
  assert.deepEqual(await runtime.run(() => fn()), { items: [{ type: 'call', id: 'c1', signature: 'sig' }] });
});
