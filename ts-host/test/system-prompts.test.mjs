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

test('withSystemPrompts scopes soft pieces over the runtime bank, nested scopes overriding', async () => {
  const { withSystemPrompts } = await import('../dist/index.js');
  const { store, port } = standIn();
  const [base, scoped, inner] = await Promise.all(['base', 'scoped', 'inner'].map(text => port.write(text)));
  const ref = block => neuraleseRef('Neuralese<SystemPrompt>', block.id);
  const answer = defineNatlang('---\nargs: { ticket: string }\nreturns: string\n---\nSay whether the ticket is urgent.\n');
  const seen = [];
  const driver = neuraleseDriver(({ messages }) => { seen.push(parts(messages[0].content)); return { calls: [['return_result', { status: 'success', value: 'yes' }]] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port, systemPrompts: systemPromptBank({ interpreter: ref(base) }) } });
  const ids = () => seen.at(-1).filter(part => part.type === 'neuralese').map(part => part.id);
  await runtime.run(() => answer('a'));
  assert.deepEqual(ids(), [base.id]);
  await withSystemPrompts({ interpreter: ref(scoped) }, () => runtime.run(() => answer('b')));
  assert.deepEqual(ids(), [scoped.id]);
  await withSystemPrompts({ interpreter: ref(scoped) }, () => withSystemPrompts({ interpreter: ref(inner) }, () => runtime.run(() => answer('c'))));
  assert.deepEqual(ids(), [inner.id]);
  await runtime.run(() => answer('d'));
  assert.deepEqual(ids(), [base.id], 'the scope ends with its function');
});

test('a large argument is listed as its digest; the variable keeps the value; small arguments stay literals', async () => {
  const { store, port } = standIn();
  const digestBlock = await port.write('a digest');
  const sites = [];
  const digest = async site => { sites.push(site); return neuraleseRef('Neuralese<Digest>', digestBlock.id); };
  const judge = defineNatlang('---\nargs: { packet: unknown, note: string }\nreturns: boolean\n---\nIs line 2 an add-on fee?\n');
  const packet = { lines: Array.from({ length: 200 }, (_, i) => ({ id: `L${i}`, text: 'x'.repeat(40) })) };
  const seen = [];
  const driver = neuraleseDriver(({ messages }) => { seen.push(messages); return { calls: [['return_result', { status: 'success', value: true }]] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port, digest } });
  assert.equal(await runtime.run(() => judge(packet, 'short')), true);
  assert.equal(sites.length, 1, 'only the large argument is digested');
  assert.equal(sites[0].name, 'packet');
  assert.equal(JSON.parse(sites[0].value).lines.length, 200, 'the digest is written from the full value');
  assert.match(sites[0].instructions, /add-on fee/, 'the receiving call\'s instructions condition the write site');
  const listing = seen[0].find(message => message.role === 'tool');
  const listed = parts(listing.content);
  assert.ok(listed.some(part => part.type === 'neuralese' && part.id === digestBlock.id), 'the listing shows the digest block');
  const text = listed.filter(part => part.type === 'text').map(part => part.text).join('');
  assert.match(text, /packet holds all of it/);
  assert.match(text, /note: string = "short"/);
  assert.doesNotMatch(text, /cut off/);
});

test('the digest instructions and listing note match the fixture the server is pinned to', async () => {
  const { readFileSync } = await import('node:fs');
  const { DIGEST_PROMPT, digestNote } = await import('../dist/native/prompt.js');
  const fixture = JSON.parse(readFileSync(new URL('../../tests/fixtures/digest-site.json', import.meta.url), 'utf8'));
  assert.equal(DIGEST_PROMPT, fixture.messages[0].content);
  assert.equal(digestNote('state'), fixture.note);
});

test('iteration guidance identifies the updated step argument as the revision source', () => {
  assert.ok(TOOLS_PROMPT.includes('latest state returned by the preceding step'));
  assert.ok(TOOLS_PROMPT.includes('preserve supported earlier edits'));
  assert.ok(TOOLS_PROMPT.includes('captured outer initial draft stays the original value'));
});


test('partial-check guidance keeps each assigned condition separate from parent aggregation', () => {
  assert.ok(TOOLS_PROMPT.includes('give each child the specific condition assigned to its evidence'));
  assert.ok(TOOLS_PROMPT.includes('Combine those checks in the parent'));
});


test('complete-record scope is preserved without joining separate records', () => {
  for (const prompt of [TOOLS_PROMPT, promptAtNlDepthLimit(TOOLS_PROMPT)]) {
    assert.ok(prompt.includes('carry its stated entity or group scope across sentences'));
    assert.ok(prompt.includes('do not require the same identifier to be repeated for every fact'));
    assert.ok(prompt.includes('Keep facts from different records separate unless the task asks for a join'));
  }
});
