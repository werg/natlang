import assert from 'node:assert/strict';
import { test } from 'node:test';
import { builtin, createNatlangRuntime, createNeuraleseLibrary, invokeAt } from '../dist/index.js';
import { builtinDefinition } from '../dist/runtime/index.js';
import { VIEW_PROMPT } from '../dist/builtin/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { isNeuraleseRef, neuraleseRef } from '../dist/native/neuralese.js';

// One summarizer family (plans/neuralese/DECISIONS.md 2026-10-09): the builtin view(value, instructions?) with a
// representation-generic result, and the query operator ask(block, question) = read(map(block, question)).

const DIALECT = 'nd:natlang@1';
const standIn = () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, DIALECT);
  // What each block was written from, so a scripted model can answer from it (the stand-in carries no meaning).
  const texts = new Map();
  const write = port.write.bind(port);
  port.write = async (text, options) => { const meta = await write(text, options); texts.set(meta.id, text); return meta; };
  return { store, port, texts };
};
const text = messages => JSON.stringify(messages);

test('view is a builtin with a representation-generic result whose body is the view prompt piece', () => {
  const definition = builtinDefinition('view');
  assert.equal(definition.generic.constraint, 'string | Neuralese<string>');
  assert.equal(definition.body.trim(), VIEW_PROMPT);
  assert.match(VIEW_PROMPT, /Without instructions, the view is faithful: keep everything needed to reproduce value exactly/);
  assert.match(VIEW_PROMPT, /With instructions, they say what the view is for: keep what that purpose needs/);
});

test('view: the crisp instance runs the body as ordinary execution, faithful without instructions and for a purpose with them', async () => {
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: 'a view' }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver });
  const view = builtin('view');
  assert.equal(await runtime.run(() => view('line 1: fee 30\nline 2: fee 40')), 'a view');
  const faithful = text(requests.at(-1).messages);
  assert.ok(faithful.includes(JSON.stringify(VIEW_PROMPT).slice(1, 40)), 'the body is the call\'s instructions');
  assert.equal(requests.at(-1).template, undefined, 'the crisp instance is not a template write');
  assert.equal(await runtime.run(() => invokeAt(view, ['line 1: fee 30', 'Only the fee of line 2.'], { kind: 'crisp' })), 'a view');
  const instructed = text(requests.at(-1).messages);
  assert.ok(instructed.includes('Only the fee of line 2.'), 'the purpose reaches the call');
  assert.ok(!faithful.includes('Only the fee of line 2.'));
});

test('view: the Neuralese instance is a template write of the same body', async () => {
  const { store, port, texts } = standIn();
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: 'fees: 30, 40' }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  const block = await runtime.run(() => invokeAt(builtin('view'), ['line 1: fee 30\nline 2: fee 40'], { kind: 'neuralese' }));
  assert.ok(isNeuraleseRef(block), JSON.stringify(block));
  assert.equal(block.$neuralese.type, 'Neuralese<string>');
  assert.equal(texts.get(block.$neuralese.id), 'fees: 30, 40');
  const request = requests.at(-1);
  assert.deepEqual([request.template.call, request.template.value], ['return_result', 'write']);
  assert.ok(text(request.messages).includes(JSON.stringify(VIEW_PROMPT).slice(1, 40)), 'the same body');
});

test('ask(block, question) is read(map(block, question)): answering from the block matches answering from its text', async () => {
  const { store, port, texts } = standIn();
  const bodies = {};
  for (const name of ['map', 'zip', 'ap', 'combine', 'split', 'splitList', 'read', 'convert', 'gloss'])
    bodies[name] = (await port.write(`the ${name} body`)).id;
  const library = { dialect: DIALECT, width: 8, bodies };
  // A scripted model that answers from what a block was written from: what ask's law asks of a trained one.
  const answer = (question, content) => question === 'Which desk?' ? /desk \d+/.exec(content)?.[0] ?? 'none' : 'unknown';
  const turns = [];
  const driver = Object.assign(async request => {
    const shown = text(request.messages);
    const blocks = [...shown.matchAll(/nz1_[a-z2-7]+/g)].map(match => match[0])
      .filter(id => texts.has(id) && !Object.values(bodies).includes(id));
    const block = blocks.at(-1);
    turns.push({ template: request.template?.value, body: Object.entries(bodies).find(([, id]) => shown.includes(id))?.[0] });
    if (request.template?.value === 'write') {
      const question = shown.includes('Which desk?') ? 'Which desk?' : '';
      return { calls: [['return_result', { status: 'success', value: answer(question, texts.get(block)) }]] };
    }
    return { calls: [['return_result', { status: 'success', value: texts.get(block) }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port }, services: { neuralese: library } });
  const note = neuraleseRef('Neuralese<string>', (await port.write('A certified copy is waiting at desk 4.')).id);
  const lib = createNeuraleseLibrary(library);
  const asked = await runtime.run(() => lib.ask(note, 'Which desk?'));
  assert.deepEqual(turns.map(turn => [turn.body, turn.template]), [['map', 'write'], ['read', 'decode']],
    'ask is map (a written answer) then read (its readout)');
  const read = await runtime.run(() => lib.read(note));
  assert.equal(asked, answer('Which desk?', read), 'ask(v, q) ≈ answering q from read(v)');
  assert.equal(asked, 'desk 4');
  await assert.rejects(async () => lib.ask(note, ''), /ask needs a question/);
  await assert.rejects(async () => lib.ask('not a block', 'Which desk?'), /ask needs a Neuralese value/);
});
