// Structural recursion of `.nl` functions (plans/semantic-fp/01 L7): a function that declares `recursive: structural`
// (or a `measure:`) sees itself by name and may call itself on a smaller argument, or where its measure falls; any other
// re-entry fails with a one-sentence error that names the fix. The same guard as TypeScript functions decides.
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore, createNatlangRuntime as createRuntime, loadNatlang } from '../dist/index.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { scriptedModel } from './support/natlang.mjs';

const TYPES = 'export type Tree = { name: string; children: Tree[] };\n';
const TREE = { name: 'a', children: [{ name: 'b', children: [{ name: 'c', children: [] }] }, { name: 'd', children: [] }] };

// Each runtime records into its own call store: verdicts and records of other runs (the machine's store) never answer.
const createNatlangRuntime = options => createRuntime({ calls: CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-'))), ...options });

function program(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-nl-recursion-'));
  writeFileSync(join(root, 'natlang.json'), '{}\n');
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}
const nlFile = (front, body) => `---\n${front}\n---\n${body}\n`;

test('a recursive: structural function calls itself on each part of its argument', async () => {
  const root = program({ 'types.ts': TYPES,
    'size.nl': nlFile('args: { tree: Tree }\nreturns: number\nrecursive: structural', 'Count the nodes of tree.') });
  const model = scriptedModel(() =>
    'const sizes = await Promise.all(tree.children.map(child => size(child))); return 1 + sizes.reduce((a, b) => a + b, 0);');
  const runtime = createNatlangRuntime({ model: model.driver, programRoot: root });
  const size = loadNatlang(join(root, 'size.nl'), root);
  assert.equal(await runtime.run(() => size(TREE)), 4);
  assert.equal(model.openings.length, 4, 'one call per node');
  // The function is listed in its own call, so the model knows it can call it.
  assert.match(model.openings[0], /size\(tree: Tree\)/);
});

test('a recursive function that calls itself on the same argument fails with the fix', async () => {
  const root = program({ 'types.ts': TYPES,
    'size.nl': nlFile('args: { tree: Tree }\nreturns: number\nrecursive: structural', 'Count the nodes of tree.') });
  const model = scriptedModel(() => 'return await size(tree);');
  const runtime = createNatlangRuntime({ model: model.driver, programRoot: root });
  const size = loadNatlang(join(root, 'size.nl'), root);
  const error = await runtime.run(() => size(TREE)).catch(error => error);
  assert.match(String(error?.message), /`size` called itself without a smaller argument\. Recurse only on a part of the input/);
});

test('a copy of a part is a smaller argument (calls receive their arguments as copies)', async () => {
  const root = program({ 'types.ts': TYPES,
    'size.nl': nlFile('args: { tree: Tree }\nreturns: number\nrecursive: structural', 'Count the nodes of tree.') });
  const model = scriptedModel(() => 'let total = 1; for (const child of tree.children) total += await size(JSON.parse(JSON.stringify(child))); return total;');
  const runtime = createNatlangRuntime({ model: model.driver, programRoot: root });
  const size = loadNatlang(join(root, 'size.nl'), root);
  assert.equal(await runtime.run(() => size(TREE)), 4);
});

test('without a recursion declaration a function cannot run below itself; the error says what to declare', async () => {
  const root = program({ 'types.ts': TYPES,
    'walk.nl': nlFile('args: { tree: Tree, again: unknown }\nreturns: number', 'Count the nodes of tree, using again for parts.') });
  const model = scriptedModel(() => 'return tree.children.length ? 1 + await again(tree.children[0], again) : 1;');
  const runtime = createNatlangRuntime({ model: model.driver, programRoot: root });
  const walk = loadNatlang(join(root, 'walk.nl'), root);
  const error = await runtime.run(() => walk(TREE, walk)).catch(error => error);
  assert.match(String(error?.message), /`walk` called itself, and its frontmatter does not declare recursion; declare `recursive: structural`/);
});

test('measure: admits a self-call where the measure falls and refuses one where it does not', async () => {
  const front = (measure) => nlFile(`args: { items: string[], done: number }\nreturns: string\nmeasure: "${measure}"`,
    'Join the items from position done on.');
  const code = 'return done >= items.length ? "" : items[done] + await join(items, done + 1);';
  const good = program({ 'join.nl': front('items.length - done') });
  let runtime = createNatlangRuntime({ model: scriptedModel(() => code).driver, programRoot: good });
  let join_ = loadNatlang(join(good, 'join.nl'), good);
  assert.equal(await runtime.run(() => join_(['a', 'b', 'c'], 0)), 'abc');

  const flat = program({ 'join.nl': front('items.length') });
  runtime = createNatlangRuntime({ model: scriptedModel(() => code).driver, programRoot: flat });
  join_ = loadNatlang(join(flat, 'join.nl'), flat);
  const error = await runtime.run(() => join_(['a', 'b'], 0)).catch(error => error);
  assert.match(String(error?.message), /`join` called itself with measure 2, which is not smaller than 2 in the call it runs below; recurse only where "items.length" falls\./);

  const negative = program({ 'join.nl': front('done - items.length') });
  runtime = createNatlangRuntime({ model: scriptedModel(() => code).driver, programRoot: negative });
  join_ = loadNatlang(join(negative, 'join.nl'), negative);
  const bad = await runtime.run(() => join_(['a'], 0)).catch(error => error);
  assert.match(String(bad?.message), /measure "done - items\.length" must be a non-negative whole number, but it is -1/);
});

test('recursion frontmatter is checked when the function loads', () => {
  const load = (front) => { const root = program({ 'f.nl': nlFile(front, 'Do it.') }); return () => loadNatlang(join(root, 'f.nl'), root); };
  assert.throws(load('args: { n: number }\nreturns: number\nrecursive: yes'), /recursive must be structural/);
  assert.throws(load('args: { n: number }\nreturns: number\nmeasure: "n +"'), /measure is a TypeScript expression over the arguments/);
  assert.throws(load('args: { n: number }\nreturns: number\nmeasure: "m - 1"'), /measure uses m, which is not an argument, an item of f\/ or a standard global/);
});

test('the system prompt states the recursion rule as the runtime applies it', () => {
  assert.match(TOOLS_PROMPT, /a function may call itself only on a smaller argument/);
  assert.doesNotMatch(TOOLS_PROMPT, /may not call themselves/);
});
