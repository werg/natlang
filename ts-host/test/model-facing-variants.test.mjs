// Model-facing variants not yet adopted (src/runtime/model-facing.ts), run with every variant on, as the `combined` arm does.
import assert from 'node:assert/strict';
import { test } from 'node:test';
process.env.NATLANG_MODEL_FACING = 'combined'; // read when the runtime modules load, so they are imported after this line
const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');
const { TOOLS_PROMPT } = await import('../dist/native/prompt.js');

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

test('the system prompt teaches inline nl with a neutral example, not the refund-policy one', () => {
  assert.match(TOOLS_PROMPT, /nl<boolean>`Is draft clear to audience\?`\(draft\)/);
  assert.match(TOOLS_PROMPT, /nl\.with<boolean>\(\{ audience \}\)`Is draft clear to audience\?`\(draft\)/);
  assert.doesNotMatch(TOOLS_PROMPT, /Refunds are allowed|Decide whether item meets policy/);
});

test('an inline nl call with no arguments and no named variable is refused', async () => {
  const { results } = await script([['eval', { code: 'const v = await nl<boolean>`Decide whether the order is late.`(); v' }]]);
  assert.match(results[0], /this nl call was given no arguments and its instructions name no variable in scope/);
});

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
