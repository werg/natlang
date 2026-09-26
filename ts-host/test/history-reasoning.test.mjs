import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { OPENING_THOUGHT } from '../dist/native/agent.js';

test('each assistant turn goes back with its reasoning, and the opening turn with a thought', async () => {
  const requests = [];
  const turns = [
    { reasoning: 'Add one to n.', calls: [['eval', { code: 'n + 1' }]] },
    { reasoning: 'It is 3.', calls: [['return_result', { status: 'success', value: 3 }]] },
  ];
  const model = async request => { requests.push(structuredClone(request.messages)); return turns[requests.length - 1]; };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { n: number }\nreturns: number\n---\nAdd one to n.\n' }, 'root.nl');
  assert.equal(await runtime.run(() => fn(2)), 3);
  const second = requests[1].filter(message => message.role === 'assistant');
  assert.equal(second[0].reasoning_content, OPENING_THOUGHT);
  assert.equal(second.at(-1).reasoning_content, 'Add one to n.');
});
