// Model-facing variants not yet adopted (src/runtime/model-facing.ts), run with every variant on, as the `combined` arm does.
import assert from 'node:assert/strict';
import { test } from 'node:test';
process.env.NATLANG_MODEL_FACING = 'combined'; // read when the runtime modules load, so they are imported after this line
const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');

test('an inline nl call that names a variable of the scope still runs', async () => {
  const seen = [];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    const opening = String(request.messages[1].content), last = request.messages.at(-1);
    const child = opening.startsWith('You are inside this call: nl@');
    if (last.role === 'tool' && last.tool_call_id !== 'scope_0') {
      seen.push(String(last.content));
      return { calls: [['return_result', { status: 'success', value: child ? true : 'done' }]] };
    }
    return { calls: [['eval', { code: 'const v = await nl<boolean>`Is name short?`(); v' }]] };
  } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nJudge name.\n' }, 'root.nl');
  assert.equal(await runtime.run(() => fn('Ada')), 'done');
  assert.ok(!seen.some(text => /was given no arguments/.test(text)), seen.join('\n---\n'));
});

test('calls after a failed call of the same reply stay in the history, each answered as not run', async () => {
  let seen;
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    const last = request.messages.at(-1);
    if (last.role === 'tool' && last.tool_call_id !== 'scope_0') {
      seen ??= request.messages;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { calls: [['eval', { code: 'throw new Error("first fails")' }], ['eval', { code: '1 + 1' }]] };
  } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  const reply = seen.findLast(message => message.role === 'assistant');
  assert.equal(reply.tool_calls.length, 2, 'both sampled calls stay in the history');
  const answers = seen.filter(message => message.role === 'tool' && reply.tool_calls.some(call => call.id === message.tool_call_id));
  assert.equal(answers.length, 2);
  assert.match(String(answers[0].content), /first fails/);
  assert.equal(String(answers[1].content), 'Not run: an earlier call in this reply failed; call it again if you still need it.');
});
