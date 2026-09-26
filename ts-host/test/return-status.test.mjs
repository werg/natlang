import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

/** Run root.nl with scripted root tool calls; returns each tool result and the call's value. */
async function script(calls) {
  const results = [];
  let step = 0;
  const model = async request => {
    if (step) results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: {}\nreturns: string\n---\nSay hello.\n' }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn()); } catch {}
  return { results, value };
}

test('an answer sent as blocked is rejected with the status that returns it', async () => {
  const { results, value } = await script([
    ['return_result', { status: 'blocked', value: 'hello' }],
    ['return_result', { status: 'success', value: 'hello' }]]);
  assert.match(results[0], /use status "success"/);
  assert.equal(value, 'hello');
});

test('a blocked call without a value is asked only for its reason', async () => {
  const { results } = await script([['return_result', { status: 'blocked' }]]);
  assert.match(results[0], /saying what is missing/);
  assert.doesNotMatch(results[0], /use status "success"/);
});

test('read_function says what a name is when it is a tool or a built-in, not a function of the program', async () => {
  const { results } = await script([['read_function', { name: 'nl' }], ['read_function', { name: 'read_page' }]]);
  assert.match(results[0], /none of its own\. nl is built into eval/);
  assert.match(results[1], /read_page is one of your tools, not a function of this program; call it directly/);
});
