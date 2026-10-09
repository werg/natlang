/**
 * Fixes the request layout that prefix caching depends on (plans/BATCHED_EXECUTION.md §3.4): messages run from stable to
 * variable. Two calls of the same function must share everything up to the arguments, and the arguments come before
 * the transcript. Changing this order is a measured decision, not a refactor.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

test('requests are laid out system, instructions and types, callable declarations, arguments, transcript', async () => {
  const outDir = mkdtempSync(join(tmpdir(), 'natlang-layout-'));
  const built = buildProject({ project: fileURLToPath(new URL('../../examples/triage', import.meta.url)), outDir, runtimeModule, writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const { triage } = await import(pathToFileURL(join(outDir, 'triage.js')).href);
  const scripted = scriptedModel(opening => {
    if (opening.includes('Pick the label')) return 'return "billing"';
    if (opening.includes('A ticket is urgent') || opening.includes('attention within the hour')) return 'return false';
    return null;
  });
  const requests = [];
  const driver = async request => { requests.push(structuredClone(request.messages)); return scripted.driver(request); };
  await createNatlangRuntime({ model: driver }).run(() => triage(['AAAA first ticket', 'BBBB second ticket'], 'billing, technical, spam'));
  const classify = requests.filter(messages => String(messages[1].content).includes('Pick the label'));
  const firsts = classify.filter(messages => messages.length === 4);   // first turn of each call
  assert.equal(firsts.length, 2);
  const [a, b] = firsts;

  // Roles: system, user (instructions and types), the pre-filled scope eval and its result, then the transcript.
  assert.deepEqual(a.map(message => message.role), ['system', 'user', 'assistant', 'tool']);
  const user = String(a[1].content);
  assert.ok(user.indexOf('You are inside this call') < user.indexOf('Instructions:'), 'signature and types precede the instructions');
  const scopeCode = JSON.parse(a[2].tool_calls[0].function.arguments).code;
  assert.ok(scopeCode.indexOf('// Functions you can call:') < 0 || scopeCode.indexOf('// Functions you can call:') < scopeCode.indexOf('read_inputs'),
    'callable declarations precede the arguments');
  assert.ok(!scopeCode.includes('AAAA'), 'argument values are not in the stable scope code');
  assert.ok(String(a[3].content).includes('AAAA'), 'argument values come in the scope result');

  // The shared prefix of two calls of one function: system, user and the scope eval; they differ first at the arguments.
  assert.deepEqual(a[0], b[0]);
  assert.deepEqual(a[1], b[1]);
  assert.deepEqual(a[2], b[2]);
  assert.notDeepEqual(a[3], b[3]);

  // Later turns only append: the transcript follows the arguments and never rewrites them.
  const later = classify.find(messages => messages.length > 4);
  assert.ok(later, 'a call took a second turn');
  assert.deepEqual(later.slice(0, 4), classify.find(messages => messages.length > 4 && String(messages[3].content) === String(later[3].content)).slice(0, 4));
  assert.ok(later.slice(4).every(message => message.role === 'assistant' || message.role === 'tool'));
});
