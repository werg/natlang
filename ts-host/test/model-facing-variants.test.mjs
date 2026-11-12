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
