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

test('read_code documents the built-ins of eval, which cannot be edited', async () => {
  const { results } = await script([['read_code', { name: 'nl' }], ['read_code', { name: 'iterateOn' }],
    ['edit_code', { name: 'nl', find: 'a', replace_with: 'b' }]]);
  assert.match(results[0], /^nl: create a natural-language function/);
  assert.match(results[1], /^iterateOn: repeat a step/);
  assert.match(results[2], /nl is built into eval and cannot be changed/);
});

test('read_code says when a name is one of the tools, not a function of the program', async () => {
  const { results } = await script([['read_code', { name: 'read_page' }]]);
  assert.match(results[0], /none of its own\. read_page is one of your tools, not a function of this program; call it directly/);
});
