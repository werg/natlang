import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { WikiWorkspace } from '../../applications/wiki.mjs';

const mergePath = fileURLToPath(new URL('../../codebases/wiki/merge_page.nl', import.meta.url));
const cellPath = fileURLToPath(new URL('../../codebases/wiki/run_cell.nl', import.meta.url));
const profile = { model: 'fixture-model', source: 'merge-v1', seed: 17 };
const page = { id: 'guide', blocks: [
  { id: 'intro', kind: 'prose', text: 'This guide explains the engine.' },
  { id: 'demo', kind: 'cell', text: 'return args.input.toUpperCase();',
    language: 'quickjs', returns: 'Text' },
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
      inputs: { base, updates, profile }, options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function merge_page(')) return { calls: [
          ['call', { function: 'prepare', to: 'let/prepared', inputs: {
            base: 'args/base', updates: 'args/updates', profile: 'args/profile' } }],
          ['call', { function: 'interpret', to: 'let/draft', inputs: {
            base: 'args/base', updates: 'let/prepared/updates' } }],
          ['call', { function: 'publish', to: 'return', inputs: {
            base: 'args/base', prepared: 'let/prepared', draft: 'let/draft',
            profile: 'args/profile' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: {
          blocks: [{ ...base.blocks[0], text: 'A small engine, with a concise example.' },
            base.blocks[1]], accounted: ['a', 'b'], unresolved: [],
        } }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'merged');
    assert.match(result.value.page.blocks[0].text, /concise example/);
    const cell = await host.run({ source: { kind: 'file', path: cellPath },
      inputs: { block_id: 'demo', input: 'hello' },
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        return { calls: [['call', { function: 'execute', to: 'return', inputs: {
          block_id: 'args/block_id', input: 'args/input' } }]], completion_tokens: 1 };
      } });
    assert.equal(cell.outcome.kind, 'done');
    assert.equal(cell.value.status, 'done');
    assert.equal(cell.value.value_text, '"HELLO"');
    assert.ok(wiki.trace('demo').length > 0);
  } finally { host.close(); }
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
    { id: 'answer', kind: 'cell', language: 'natlang', returns: 'Text',
      text: 'Return a short answer to input.' },
  ] }, { profile, modelTurn: async () => {
    if (turns++ > 0) return { calls: [], text: 'done', completion_tokens: 1 };
    entered(); await gate;
    return { calls: [['write', { path: 'return', value: 'Old answer' }]],
      completion_tokens: 1 };
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
