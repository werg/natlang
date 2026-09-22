import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compileScopeSnippet, SCOPE_COMPILE_VERSION } from '../dist/index.js';

function load(result) {
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics));
  return Function(`${result.program}\nreturn ${result.entrypoint};`)();
}

test('scope compiler records top-level bindings and preserves final-expression REPL semantics', async () => {
  const compiled = compileScopeSnippet(
    'const selected: Num = input + 1;\nlet label = `v${selected}`;\nlabel',
    { inputBindings: ['input'] });
  assert.equal(compiled.version, SCOPE_COMPILE_VERSION);
  assert.deepEqual(compiled.bindings.map(({ name, kind, mutable, annotation }) =>
    ({ name, kind, mutable, annotation })), [
    { name: 'selected', kind: 'const', mutable: false, annotation: 'Num' },
    { name: 'label', kind: 'let', mutable: true, annotation: undefined },
  ]);
  assert.deepEqual(compiled.finalExpression, { start: 61, end: 66, line: 3, column: 1 });
  assert.match(compiled.body, /return __natlang_finish\(label\);/);
  const run = load(compiled);
  assert.deepEqual(await run({ input: 4 }, {}, {}),
    { result: 'v5', bindings: { selected: 5, label: 'v5' } });
});

test('scope compiler injects async checked-helper placeholders with ordinary call syntax', async () => {
  const compiled = compileScopeSnippet('const answer = await double(value);\nanswer', {
    inputBindings: ['value'], helperBindings: ['double'],
  });
  const calls = [];
  const run = load(compiled);
  assert.deepEqual(await run({ value: 7 }, {}, async (name, args) => {
    calls.push([name, args]); return args[0] * 2;
  }),
    { result: 14, bindings: { answer: 14 } });
  assert.deepEqual(calls, [['double', [7]]]);
});

test('scope compiler preserves explicit return statements', async () => {
  const compiled = compileScopeSnippet('if (flag) return 3;\nreturn 5;', { inputBindings: ['flag'] });
  assert.equal(compiled.finalExpression, undefined);
  const run = load(compiled);
  assert.deepEqual(await run({ flag: true }, {}, {}), { result: 3, bindings: {} });
  assert.deepEqual(await run({ flag: false }, {}, {}), { result: 5, bindings: {} });
});

test('scope compiler reports forbidden capabilities with original source spans', () => {
  const source = 'const a = await import("node:fs");\ntry { fetch("https://example.test"); } catch {}';
  const compiled = compileScopeSnippet(source);
  assert.equal(compiled.ok, false);
  assert.equal(compiled.program, undefined);
  assert.ok(compiled.diagnostics.some(item => item.code === 'forbidden-dynamic-code' && item.line === 1));
  assert.ok(compiled.diagnostics.some(item => item.code === 'forbidden-control' && item.line === 2));
  assert.ok(compiled.diagnostics.some(item => item.code === 'forbidden-ambient' && item.line === 2));
});

test('scope compiler rejects prototype mutation but permits matching data keys', () => {
  const data = compileScopeSnippet('const record = { process: "data", "__proto__": "value" };\nrecord.process');
  assert.equal(data.ok, true, JSON.stringify(data.diagnostics));
  const mutation = compileScopeSnippet('record.__proto__ = other;');
  assert.deepEqual(mutation.diagnostics.map(item => item.code), ['forbidden-prototype-mutation']);
});

test('scope compiler reports syntax errors and injected-binding collisions', () => {
  const syntax = compileScopeSnippet('const value = ;');
  assert.equal(syntax.ok, false);
  assert.ok(syntax.diagnostics.some(item => item.code === 'typescript-syntax'));
  const collision = compileScopeSnippet('const value = 1; value', { inputBindings: ['value'] });
  assert.equal(collision.ok, false);
  assert.ok(collision.diagnostics.some(item => item.code === 'invalid-binding'));
});

test('scope compiler transactionally captures existing locals and rejects immutable writes', async () => {
  const compiled = compileScopeSnippet('count += step;\nconst doubled = count * 2;\ndoubled', {
    inputBindings: ['step'], localBindings: [{ name: 'count', mutable: true }],
  });
  const run = load(compiled);
  assert.deepEqual(await run({ step: 3 }, { count: 4 }, {}),
    { result: 14, bindings: { count: 7, doubled: 14 } });

  const immutable = compileScopeSnippet('settings = { enabled: false };', {
    localBindings: [{ name: 'settings', mutable: false }],
  });
  assert.equal(immutable.ok, false);
  assert.ok(immutable.diagnostics.some(item => item.code === 'invalid-binding'));
});

test('scope compiler rejects an early return that would skip persistent declaration initialization', () => {
  const compiled = compileScopeSnippet('if (stop) return 1;\nconst later = 2;\nlater', { inputBindings: ['stop'] });
  assert.equal(compiled.ok, false);
  assert.ok(compiled.diagnostics.some(item => item.code === 'forbidden-control' && /atomically/.test(item.message)));
});
