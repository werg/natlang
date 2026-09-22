import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NatlangHost, NodeFileTree } from '../dist/index.js';
import { WikiWorkspace } from '../../applications/wiki.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const mergePath = fileURLToPath(new URL('../../codebases/wiki/merge_page.nl', import.meta.url));
const cellPath = fileURLToPath(new URL('../../codebases/wiki/run_cell.nl', import.meta.url));
const profile = { model: 'fixture-model', source: 'merge-v1', seed: 17 };
const page = { id: 'guide', blocks: [
  { id: 'intro', kind: 'prose', text: 'This guide explains the engine.' },
  { id: 'demo', kind: 'cell', text: 'return input.toUpperCase();',
    language: 'quickjs', returns: 'string' },
] };

test('natlang merges two page edits and runs a pinned, limited child cell', async () => {
  const wiki = new WikiWorkspace(page, { profile });
  const base = wiki.snapshot();
  const updates = [
    { id: 'b', block_id: 'intro', base_revision: base.revision,
      author: 'bob', text: 'Add a concise example.' },
    { id: 'a', block_id: 'intro', base_revision: base.revision,
      author: 'alice', text: 'Clarify that the engine is small.' },
  ];
  const host = new NatlangHost({ host: { wiki,
    drainEvents: () => wiki.drainEvents() } });
  try {
    const result = await host.run({ source: { kind: 'file', path: mergePath },
      inputs: { base, updates, profile },
      modelTurn: turn => {
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function merge_page(')) return evalTurn(turn,
          'const prepared = await prepare(base, updates, profile); if (!prepared.valid) return reject(base, prepared); const draft = await interpret(base, prepared.updates); await publish(base, prepared, draft, profile)');
        return evalTurn(turn, `(${JSON.stringify({
          blocks: [{ ...base.blocks[0], text: 'A small engine, with a concise example.' },
            base.blocks[1]], accounted: ['a', 'b'], unresolved: [],
        })})`);
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'merged');
    assert.match(result.value.page.blocks[0].text, /concise example/);
    const cell = await host.run({ source: { kind: 'file', path: cellPath },
      inputs: { block_id: 'demo', input: 'hello' },
      modelTurn: turn => evalTurn(turn, 'await execute(block_id, input)') });
    assert.equal(cell.outcome.kind, 'done');
    assert.equal(cell.value.status, 'done');
    assert.equal(cell.value.value_text, '"HELLO"');
    assert.ok(wiki.trace('demo').length > 0);
  } finally { host.close(); }
});

test('wiki binds files only to natlang cells and refreshes the provider per run', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-wiki-files-'));
  writeFileSync(join(folder, 'note.txt'), 'first note');
  const wiki = new WikiWorkspace({ id: 'files', blocks: [
    { id: 'quick', kind: 'cell', language: 'quickjs', returns: 'string', text: 'return input;' },
    { id: 'natural', kind: 'cell', language: 'natlang', returns: 'string', text: 'Read the named project note and return its text.' },
  ] }, { profile, files: () => new NodeFileTree(folder),
    modelTurn: request => evalTurn(request, 'files["note.txt"].text') });
  try {
    assert.equal((await wiki.runCell('quick', 'quick value')).value_text, '"quick value"');
    assert.equal((await wiki.runCell('natural', 'question')).value_text, '"first note"');
    writeFileSync(join(folder, 'note.txt'), 'second note');
    assert.equal((await wiki.runCell('natural', 'question')).value_text, '"second note"');
  } finally { wiki.close?.(); rmSync(folder, { recursive: true, force: true }); }
});

test('mismatched profiles, missing updates and invalid merged cells are rejected', async () => {
  const wiki = new WikiWorkspace(page, { profile });
  const base = wiki.snapshot();
  const update = { id: 'x', block_id: 'intro', base_revision: base.revision,
    author: 'editor', text: 'Clarify source.' };
  assert.equal(wiki.prepare(base, [update], { ...profile, seed: 18 }).valid, false);
  const prepared = wiki.prepare(base, [update], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: base.blocks, accounted: [],
    unresolved: [] }, profile).status, 'rejected');
  assert.equal(wiki.publish(base, prepared, { blocks: [base.blocks[0],
    { ...base.blocks[1], returns: 'Any' }], accounted: ['x'], unresolved: [] }, profile).status,
  'rejected');
  const good = wiki.publish(base, prepared, { blocks: [
    { ...base.blocks[0], text: 'Clarified source.' }, base.blocks[1]],
    accounted: ['x'], unresolved: [] }, profile);
  assert.equal(good.status, 'merged');
  assert.equal(wiki.prepare(base, [update], profile).valid, false);
});

test('late natlang cell completion cannot replace output from a newer page', async () => {
  let release, entered;
  let turns = 0;
  const started = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const wiki = new WikiWorkspace({ id: 'live', blocks: [
    { id: 'note', kind: 'prose', text: 'Old page' },
    { id: 'answer', kind: 'cell', language: 'natlang', returns: 'string',
      text: 'Return a short answer to input.' },
  ] }, { profile, modelTurn: async request => {
    turns++;
    entered(); await gate;
    return evalTurn(request, '"Old answer"');
  } });
  const running = wiki.runCell('answer', 'question');
  await started;
  const base = wiki.snapshot();
  const update = { id: 'u1', block_id: 'note', base_revision: base.revision,
    author: 'editor', text: 'New page' };
  const prepared = wiki.prepare(base, [update], profile);
  assert.equal(wiki.publish(base, prepared, { blocks: [
    { ...base.blocks[0], text: 'New page' }, base.blocks[1]],
    accounted: ['u1'], unresolved: [] }, profile).status, 'merged');
  release();
  const result = await running;
  assert.equal(result.status, 'stale');
  assert.equal(wiki.output('answer'), null);
});
