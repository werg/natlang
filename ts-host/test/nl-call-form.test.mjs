import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { desugarNlCalls } from '../dist/compiler/nl-call.js';

/** Root evals `code` then returns what it evaluated to; any nl child call answers `answer`. */
async function run(code, answer) {
  const outputs = [];
  let root = 0;
  const model = async request => {
    const opening = String(request.messages[1].content);
    if (!opening.includes('root(')) return { calls: [['return_result', { status: 'success', value: answer }]] };
    root++;
    if (root === 1) return { calls: [['eval', { code }]] };
    outputs.push(String(request.messages.at(-1).content));
    return { calls: [['return_result', { status: 'success', value: 0 }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { n: number }\nreturns: number\n---\nAdd one to n.\n' }, 'root.nl');
  await runtime.run(() => fn(2));
  return outputs[0];
}

test('nl called like a function on its instructions is a one-shot call whose result is the answer', async () => {
  assert.match(await run('const answer: number = await nl(`Add one to ${n}.`);\nanswer', 3), /^3\b/);
  assert.match(await run('const answer = await nl<number>("Add one to n.");\nanswer', 3), /^3\b/);
});

test('the rewrite keeps positions, and instructions built at run time are reported with the form that works', async () => {
  const written = 'const a = await nl(`Is ${x} big?`), b = nl<boolean>(\'Is x big?\');';
  assert.equal(desugarNlCalls(written).length, written.length);
  assert.equal(desugarNlCalls(written), 'const a = await nl`Is ${x} big?`(), b = nl<boolean>`Is x big?`();');
  const output = await run('const q = "Add one to n.";\nconst answer: number = await nl(q);\nanswer', 3);
  assert.match(output, /nl is a template tag, not a function taking a string/);
});

test('an nl result nothing types runs open, shaped by the fields the code reads', async () => {
  assert.match(await run('const r = await nl`Read the amount in n.`(n);\nr.amount + 1', { amount: 2 }), /^3\b/);
  assert.match(await run('const r = await nl`Say something about n.`(n);\nr', 'anything at all'), /anything at all/);
});
