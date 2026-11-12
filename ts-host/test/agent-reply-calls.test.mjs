// Decided model-facing behaviour (2026-11-12, plans/MODEL_FACING_CHANGES.md): every sampled call of a reply stays in the
// history, and every turn offers the same tools.
import assert from 'node:assert/strict';
import { test } from 'node:test';
const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');

test('calls after a failed call of the same reply stay in the history, each answered as not run', async () => {
  let seen;
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    const last = request.messages.at(-1);
    if (last.role === 'tool' && last.tool_call_id !== 'scope_0') {
      seen ??= request.messages;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { calls: [['eval', { code: 'throw new Error("first fails")' }], ['eval', { code: '1 + 1' }],
      ['list_files', {}]] };
  } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  const reply = seen.findLast(message => message.role === 'assistant');
  assert.equal(reply.tool_calls.length, 3, 'every sampled call stays in the history');
  const answers = seen.filter(message => message.role === 'tool' && reply.tool_calls.some(call => call.id === message.tool_call_id));
  assert.equal(answers.length, 3);
  assert.match(String(answers[0].content), /first fails/);
  assert.equal(String(answers[1].content), 'Not run: an earlier call in this reply failed; call it again if you still need it.');
  assert.doesNotMatch(String(answers[2].content), /^Not run/, 'a read-only call still runs');
});

test('with a fixed tool list the last turn still offers every tool and rejects any call but return_result', async () => {
  const offered = [], answers = [];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: { maxTurns: 2, driver: async request => {
    offered.push(request.tools.map(tool => tool.function.name));
    const last = request.messages.at(-1);
    if (last.role === 'tool' && last.tool_call_id !== 'scope_0') answers.push(String(last.content));
    return offered.length === 1 ? { calls: [['eval', { code: '1 + 1' }]] } : { calls: [['eval', { code: '2 + 2' }]] };
  } } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  assert.ok(offered.length >= 2 && offered.every(names => names.join() === offered[0].join()), JSON.stringify(offered));
});
