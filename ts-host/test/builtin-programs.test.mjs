import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { builtin, builtinDefinition, builtinNames, builtinRecords } from '../dist/runtime/index.js';
import { BUILTIN_SOURCES, builtinBody, builtinSource } from '../dist/builtin/index.js';
import { renderBuiltinPrompts } from './support/builtin-prompts.mjs';

const folder = new URL('../src/builtin/', import.meta.url).pathname;

test('what the model sees for the built-in natlang functions is byte-identical to the base commit', async () => {
  const before = JSON.parse(readFileSync(new URL('./fixtures/builtin-prompts.json', import.meta.url), 'utf8'));
  const after = JSON.parse(JSON.stringify(await renderBuiltinPrompts()));
  for (const key of Object.keys(before)) assert.deepEqual(after[key], before[key], `${key} changed what the model sees`);
  assert.deepEqual(Object.keys(after).sort(), Object.keys(before).sort());
  // Prompt-piece ids and texts, which soft-prompt substitution and the training-data converter match textually.
  assert.deepEqual(after.promptPieces.map(piece => piece.id), before.promptPieces.map(piece => piece.id));
  for (const id of ['progress-judge', 'digest'])
    assert.equal(after.promptPieces.find(piece => piece.id === id).text, before.promptPieces.find(piece => piece.id === id).text);
});

test('the embedded sources are the .nl files of src/builtin', () => {
  const files = readdirSync(folder).filter(name => name.endsWith('.nl')).map(name => name.slice(0, -3)).sort();
  assert.deepEqual(Object.keys(BUILTIN_SOURCES).sort(), files);
  assert.deepEqual(builtinNames().sort(), files);
  for (const name of files) assert.equal(builtinSource(name), readFileSync(join(folder, `${name}.nl`), 'utf8'));
  assert.throws(() => builtinSource('nothing'), /no built-in natlang program named "nothing"/);
});

test('built-ins load as a callable folder and resolve by name', () => {
  const records = builtinRecords();
  for (const name of builtinNames()) {
    assert.equal(records[name]?.kind, 'natlang', name);
    assert.equal(records[name].instructions, `${builtinBody(name)}\n`);
    assert.equal(typeof builtin(name), 'function');
    assert.equal(builtin(name), builtin(name));
  }
  assert.throws(() => builtin('nothing'), /no built-in natlang program named "nothing"/);
  assert.throws(() => builtin('constructor'), /no built-in natlang program/);
  // The runtime's own calls keep the identity the literals had.
  assert.deepEqual([builtinDefinition('progressJudge').id, builtinDefinition('progressJudge').name], ['natlang:progress_judge', 'progress_judge']);
  assert.deepEqual([builtinDefinition('compareBehaviors').id, builtinDefinition('compareBehaviors').name], ['natlang:compareBehaviors', 'compareBehaviors']);
});

test('the built-ins ship in the build and in the staged npm package', () => {
  assert.ok(existsSync(new URL('../dist/builtin/sources.generated.js', import.meta.url)));
  const staged = new URL('../../npm-packages/node/dist/builtin/sources.generated.js', import.meta.url);
  if (existsSync(new URL('../../npm-packages/node/dist/index.js', import.meta.url))) assert.ok(existsSync(staged), 'the node package stages the built-ins');
});
