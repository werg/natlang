/**
 * A host or TypeScript function is a directory reducer: `folder.apply`, `folder.propose` and `folder.iterateOn` take a
 * plain function `(folder, ...args)`. It runs on a private transaction copy of the folder, as a natural-language reducer
 * does: its changes are installed when it returns and discarded when it throws.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, createNatlangRuntime, IterationLimitError } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const runtime = () => createNatlangRuntime({ model: async () => { throw Error('unexpected model request'); } });

test('folder.apply runs a function on a private copy and installs its changes when it returns', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': 'before' });
  const value = await runtime().run(() => folder.apply(async (draft, suffix) => {
    await draft.file('entry.ts').writeText('after ' + suffix);
    assert.equal(await folder.readText('entry.ts'), 'before', 'the caller sees nothing until the function returns');
    return 'done';
  }, 'edit'));
  assert.equal(value, 'done');
  assert.equal(await folder.readText('entry.ts'), 'after edit');
});

test('a function that throws leaves the folder as it was', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': 'before' });
  await assert.rejects(() => runtime().run(() => folder.apply(async draft => { await draft.file('entry.ts').writeText('partial'); throw Error('refused'); })), /refused/);
  assert.equal(await folder.readText('entry.ts'), 'before');
});

test('folder.propose with a function yields a proposal and leaves the folder unchanged until accepted', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': 'before' });
  const proposal = await runtime().run(() => folder.propose(async draft => { await draft.file('entry.ts').writeText('candidate'); return { summary: 'edited' }; }));
  assert.deepEqual(proposal.value, { summary: 'edited' });
  assert.equal(await folder.readText('entry.ts'), 'before');
  assert.equal(await proposal.folder.readText('entry.ts'), 'candidate');
  await folder.accept(proposal);
  assert.equal(await folder.readText('entry.ts'), 'candidate');
});

test('folder.iterateOn takes a function step: joint checkpoints, measure, rollback of a failed step', async () => {
  const folder = Folder.fromFiles({ 'entry.ts': '0' });
  const step = async (draft, state, increment) => { await draft.file('entry.ts').writeText(String(state.n + increment)); return { n: state.n + increment }; };
  const result = await runtime().run(() => folder.iterateOn(step, { n: 0 }, 1).withMeasure(state => 3 - state.n).until(state => state.n === 3));
  assert.equal(await result.folder.readText('entry.ts'), '3');
  assert.equal(await folder.readText('entry.ts'), '0', 'the receiving folder is unchanged');
  assert.deepEqual(result.state, { n: 3 });
  const failing = async draft => { await draft.file('entry.ts').writeText('failed'); throw Error('bad step'); };
  let failure;
  await assert.rejects(() => runtime().run(() => result.folder.iterateOn(failing, result.state).withLimit({ maxSteps: 1 }).checkProgress('off').until(() => false)), error => { failure = error; return true; });
  assert.equal(failure.lastState.folder.digest, result.folder.digest);
  assert.equal(await failure.lastState.folder.readText('entry.ts'), '3');
  await assert.rejects(() => runtime().run(() => folder.iterateOn(step, { n: 0 }, 1).checkProgress('off').withLimit({ maxSteps: 1 }).until(() => false)), IterationLimitError);
});

test('a function reducer can apply a natural-language reducer inside its transaction', async () => {
  const files = { 'edit.nl': '---\nkind: directory-reducer\nargs: {}\nreturns: string\n---\nWrite entry.ts.' };
  const edit = loadVirtualNatlang(files, 'edit.nl');
  const model = async () => ({ calls: [['eval', { code: 'await folder.file("entry.ts").writeText("by the model"); return "wrote";', finish: true }]] });
  const folder = Folder.fromFiles({ 'entry.ts': 'before' });
  const task = createNatlangRuntime({ model });
  const wrapper = async draft => {
    const summary = await draft.apply(edit);
    return { summary, text: await draft.file('entry.ts').readText() };
  };
  assert.deepEqual(await task.run(() => folder.apply(wrapper)), { summary: 'wrote', text: 'by the model' });
  assert.equal(await folder.readText('entry.ts'), 'by the model');
  const refusing = async draft => { await draft.apply(edit); throw Error('checked and refused'); };
  const other = Folder.fromFiles({ 'entry.ts': 'before' });
  await assert.rejects(() => createNatlangRuntime({ model }).run(() => other.apply(refusing)), /checked and refused/);
  assert.equal(await other.readText('entry.ts'), 'before', 'the nested edit is discarded with its caller');
});

test('a natural-language function that is not a directory reducer is refused, not called with a folder', async () => {
  const plain = loadVirtualNatlang({ 'plain.nl': '---\nargs: {}\nreturns: string\n---\nReturn a word.' }, 'plain.nl');
  await assert.rejects(() => runtime().run(() => Folder.fromFiles({}).apply(plain)), /directory reducer/);
  await assert.rejects(() => runtime().run(() => Folder.fromFiles({}).apply('not a reducer')), /directory reducer/);
});
