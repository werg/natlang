import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang, findPieces, promptPieces, softenText, systemPromptBank } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { TOOLS_PROMPT, promptAtNlDepthLimit } from '../dist/native/prompt.js';

const standIn = () => {
  const store = new MemoryNeuraleseStore();
  return { store, port: new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1') };
};
const neuraleseDriver = fn => Object.assign(fn, { neuralese: true });
const parts = content => Array.isArray(content) ? content : [{ type: 'text', text: String(content ?? '') }];

test('prompt pieces have stable unique IDs and distinct texts', () => {
  const pieces = promptPieces();
  assert.equal(new Set(pieces.map(p => p.id)).size, pieces.length);
  assert.equal(new Set(pieces.map(p => p.text)).size, pieces.length);
  for (const id of ['interpreter', 'interpreter/depth-limit', 'function-tools', 'decision', 'compaction-notice', 'handover/open', 'predicate/0'])
    assert.ok(pieces.some(p => p.id === id), id);
});

test('softening replaces banked pieces longest first and leaves other text alone', async () => {
  const { port } = standIn();
  const soft = {}, refs = {};
  for (const id of ['interpreter', 'interpreter/depth-limit']) {
    refs[id] = (await port.write(id)).id;
    soft[id] = neuraleseRef('Neuralese<SystemPrompt>', refs[id]);
  }
  const bank = systemPromptBank(soft);
  const plain = softenText(TOOLS_PROMPT + '\n\nProgram guidance stays text.', bank);
  assert.ok(plain.includes(refs.interpreter) && plain.endsWith('Program guidance stays text.'));
  assert.ok(!plain.includes('You are running one call'));
  const limited = softenText(promptAtNlDepthLimit(TOOLS_PROMPT), bank);
  assert.ok(limited.includes(refs['interpreter/depth-limit']) && !limited.includes(refs.interpreter), 'the variant wins');
  assert.equal(findPieces('nothing to see', promptPieces()).length, 0);
  assert.throws(() => systemPromptBank({ 'no-such-piece': soft.interpreter }), /system-prompt-unknown-piece/);
});

test('a Neuralese driver gets the banked system prompt as a block; a crisp driver gets the text', async () => {
  const { store, port } = standIn();
  const block = await port.write('the interpreter prompt');
  const bank = systemPromptBank({ interpreter: neuraleseRef('Neuralese<SystemPrompt>', block.id) });
  const answer = defineNatlang('---\nargs: { ticket: string }\nreturns: string\n---\nSay whether the ticket is urgent.\n');

  const soft = [];
  const driver = neuraleseDriver(({ messages }) => { soft.push(messages); return { calls: [['return_result', { status: 'success', value: 'yes' }]] }; });
  assert.equal(await createNatlangRuntime({ model: driver, neuralese: { store, port, systemPrompts: bank } }).run(() => answer('down')), 'yes');
  const system = parts(soft[0][0].content);
  assert.ok(system.some(part => part.type === 'neuralese' && part.id === block.id), 'the system prompt is the block');
  assert.ok(!system.some(part => part.type === 'text' && part.text.includes('You are running one call')));

  const crisp = [];
  const plain = ({ messages }) => { crisp.push(messages); return { calls: [['return_result', { status: 'success', value: 'yes' }]] }; };
  assert.equal(await createNatlangRuntime({ model: plain, neuralese: { store, port, systemPrompts: bank } }).run(() => answer('down')), 'yes');
  assert.ok(String(crisp[0][0].content).includes('You are running one call'), 'a crisp driver keeps the text');
});
