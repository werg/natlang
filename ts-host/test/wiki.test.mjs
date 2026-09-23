import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { WikiWorkspace } from '../../applications/dist/wiki/index.js';
import { scriptedModel } from './support/natlang.mjs';

const profile = { model: 'fixture-model', source: 'merge-v1', seed: 17 };
const page = { id: 'guide', blocks: [
  { id: 'intro', kind: 'prose', text: 'This guide explains the engine.' },
  { id: 'demo', kind: 'cell', text: 'return input.toUpperCase();', language: 'javascript', returns: 'string' },
] };
const wikiRuntime = model => createNatlangRuntime({ model: model.driver, seed: { mode: 'derived', root: profile.seed } });

test('natlang reconciles two page edits and a JavaScript cell runs against the merged page', async () => {
  const model = scriptedModel(opening => {
    assert.match(opening, /Reconcile the meaning of the updates/);
    return 'return { blocks: [{ ...base.blocks[0], text: "A small engine, with a concise example." }, base.blocks[1]], ' +
      'accounted: updates.map(update => update.id), unresolved: [] }';
  });
  const wiki = new WikiWorkspace(page, { profile, runtime: wikiRuntime(model) });
  const base = wiki.snapshot();
  const report = await wiki.merge(base, [
    { id: 'b', block_id: 'intro', base_revision: base.revision, author: 'bob', text: 'Add a concise example.' },
    { id: 'a', block_id: 'intro', base_revision: base.revision, author: 'alice', text: 'Clarify that the engine is small.' }]);
  assert.equal(report.status, 'merged', report.detail);
  assert.match(report.page.blocks[0].text, /concise example/);
  assert.deepEqual(wiki.drainEvents().map(event => event.operation), ['wiki.merge']);
  const cell = await wiki.runCell('demo', 'hello');
  assert.equal(cell.status, 'done');
  assert.equal(cell.value_text, '"HELLO"');
});

test('natlang cells get the project folder, read fresh for each run', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-wiki-files-'));
  writeFileSync(join(folder, 'note.txt'), 'first note');
  const model = scriptedModel(() => 'return (await files.file("note.txt").readText())');
  const wiki = new WikiWorkspace({ id: 'files', blocks: [
    { id: 'quick', kind: 'cell', language: 'javascript', returns: 'string', text: 'return input;' },
    { id: 'natural', kind: 'cell', language: 'natlang', returns: 'string', text: 'Read the project note and return its text.' },
  ] }, { profile, runtime: wikiRuntime(model), files: () => openFolder(folder).root() });
  try {
    assert.equal((await wiki.runCell('quick', 'quick value')).value_text, '"quick value"');
    assert.equal(model.openings.length, 0, 'JavaScript cells do not call the model');
    const first = await wiki.runCell('natural', 'question');
    assert.equal(first.value_text, '"first note"');
    assert.ok(first.trace_events > 0 && wiki.trace('natural').length === 1);
    writeFileSync(join(folder, 'note.txt'), 'second note');
    assert.equal((await wiki.runCell('natural', 'question')).value_text, '"second note"');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('mismatched profiles, missing updates and invalid merged cells are rejected', () => {
  const wiki = new WikiWorkspace(page, { profile, runtime: createNatlangRuntime() });
  const base = wiki.snapshot();
  const update = { id: 'x', block_id: 'intro', base_revision: base.revision, author: 'editor', text: 'Clarify source.' };
  assert.equal(wiki.prepare(base, [update], { ...profile, seed: 18 }).valid, false);
  const prepared = wiki.prepare(base, [update], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: base.blocks, accounted: [], unresolved: [] }, profile).status, 'rejected');
  assert.equal(wiki.publish(base, prepared, { blocks: [base.blocks[0], { ...base.blocks[1], returns: 'Any' }],
    accounted: ['x'], unresolved: [] }, profile).status, 'rejected');
  const good = wiki.publish(base, prepared, { blocks: [{ ...base.blocks[0], text: 'Clarified source.' }, base.blocks[1]],
    accounted: ['x'], unresolved: [] }, profile);
  assert.equal(good.status, 'merged');
  assert.equal(wiki.prepare(base, [update], profile).valid, false, 'the old base is stale after a merge');
});

test('a natlang cell that finishes after the page changed is stale and not shown', async () => {
  let release, entered;
  const started = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const model = scriptedModel(async () => { entered(); await gate; return 'return "Old answer"'; });
  const wiki = new WikiWorkspace({ id: 'live', blocks: [
    { id: 'note', kind: 'prose', text: 'Old page' },
    { id: 'answer', kind: 'cell', language: 'natlang', returns: 'string', text: 'Return a short answer to input.' },
  ] }, { profile, runtime: wikiRuntime(model) });
  const running = wiki.runCell('answer', 'question');
  await started;
  const base = wiki.snapshot();
  const prepared = wiki.prepare(base, [{ id: 'u1', block_id: 'note', base_revision: base.revision, author: 'editor', text: 'New page' }], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: [{ ...base.blocks[0], text: 'New page' }, base.blocks[1]],
    accounted: ['u1'], unresolved: [] }, profile).status, 'merged');
  release();
  assert.equal((await running).status, 'stale');
  assert.equal(wiki.output('answer'), null);
});
