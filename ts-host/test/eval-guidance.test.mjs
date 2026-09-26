import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { BUILT_INS_LINE } from '../dist/native/agent.js';

/** Run a call of greet(name) with scripted root calls; returns the opening, each tool result, and the value. */
async function script(calls, returns = 'string') {
  const results = [];
  let opening = '', step = 0;
  const model = async request => {
    if (!opening) opening = String(request.messages[1].content); else results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': `---\nargs: { name: string }\nreturns: ${returns}\n---\nGreet name.\n` }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn('Ada')); } catch {}
  return { opening, results, value };
}

test('the opening names the built-ins and where their documentation is', async () => {
  const { opening } = await script([['return_result', { status: 'success', value: 'hi' }]]);
  assert.ok(opening.includes(BUILT_INS_LINE));
});

test("return_result in eval with the tool's shape is read as the tool's request", async () => {
  const { results } = await script([['eval', { code: 'return_result({ status: "success", value: "Hello, " + name })' }]]);
  assert.match(results[0], /Staged "Hello, Ada" as the result/);
  const blocked = await script([['eval', { code: 'return_result({ status: "blocked", reason: "No greeting style is given." })' }]]);
  assert.deepEqual(blocked.results, [], 'blocked ends the call');
});

test('a value of the return type shaped like a request stays a value', async () => {
  const { results } = await script([['eval', { code: 'return_result({ status: "success", value: 1 })' }]],
    '"{ status: string, value: number }"');
  assert.match(results[0], /Staged \{ status: "success", value: 1 \} as the result/);
});

test('reaching for what the scope lacks is answered with what it has', async () => {
  const { results } = await script([['eval', { code: 'const fs = require("fs");\nfs' }], ['eval', { code: 'missingThing + 1' }]]);
  for (const text of results) assert.match(text, /This call's eval scope has name, the built-ins nl, iterateOn and transcript/);
});

test("redeclaring an input says it already holds the caller's value", async () => {
  const { results } = await script([['eval', { code: 'const name = "Test";\nname' }]]);
  assert.match(results[0], /name is this call's input and already holds the caller's value; use it directly/);
});

test('Node\'s own modules cannot be imported in eval; they are not packages', async () => {
  const { results } = await script([['eval', { code: "import * as fs from 'fs';\nfs.readdirSync('/')" }],
    ['eval', { code: "const cp = await import('child_process');\ncp" }]]);
  for (const text of results) assert.match(text, /is part of Node, not a package: eval code works with this call's scope/);
  assert.ok(results.every(text => !text.includes('bin')), 'no directory listing came back');
});
