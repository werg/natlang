import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { extractFunctions } from '../scripts/code-corpus/extract.mjs';
import { instrumentSource } from '../scripts/code-corpus/capture.mjs';

test('extracts documented declarations with original defaults, types, docs, and free binding assessment', () => {
  const source = `/** Adds a value. */\nexport function add(x: number = 3, y: number): number { const z = x + y; return z; }\nfunction hidden(v) { return v; }\nfunction uses(v) { return helper(v); }`;
  const records = extractFunctions(source, { path: 'math.ts', sourceName: 'fixture', revision: 'abc', license: 'MIT' });
  assert.equal(records.length, 1);
  assert.equal(records[0].function.name, 'add');
  assert.equal(records[0].instruction, 'Adds a value.');
  assert.equal(records[0].function.parameters[0].type, 'number');
  assert.match(records[0].function.source, /x: number = 3/);
  assert.deepEqual(records[0].verification, { status: 'unverified' });
  const override = extractFunctions('function f(x) { return sibling(x); }', { instruction: 'Use sibling.', path: 'x.js' });
  assert.match(override[0].verification.reasons[0], /sibling/);
  const imported = extractFunctions('import { helper } from "./helper.js";\n/** Transform a value. */\nexport function f(x: number) { return helper(x); }', { path:'src/main.ts' });
  assert.deepEqual(imported[0].function.imports, [{specifier:'./helper.js',source:'import { helper } from "./helper.js";'}]);
  assert.match(imported[0].function.body, /helper\(x\)/);
});

test('extractor flags direct and transitive recursion before corpus replay', () => {
  const direct = extractFunctions('/** Recurse. */ export function f(x: number): number { return f(x - 1); }', { path: 'f.ts' });
  assert.equal(direct[0].function.recursive, true);
  assert.match(direct[0].verification.reasons.join(' '), /recursive function graph/);
  const indirect = extractFunctions('/** Recurse indirectly. */ export function f(x: number): number { return g(x); }\nfunction g(x: number): number { return f(x); }', { path: 'f.ts' });
  assert.equal(indirect[0].function.recursive, true);
  const safe = extractFunctions('/** Double. */ export function f(x: number): number { return g(x); }\nfunction g(x: number): number { return x * 2; }', { path: 'f.ts' });
  assert.equal(safe[0].function.recursive, false);
});

test('instrumented module captures defaults, input mutation, returns, and throws in child process', () => {
  const dir = mkdtempSync(join(tmpdir(), 'code-corpus-'));
  try {
    const source = `export function mutate(value, amount = 2) { value.count += amount; return value.count; }\nexport function fail(value) { value.count++; throw new Error('bad input'); }`;
    const transformed = instrumentSource(source, { path: 'fixture.js', sourceName: 'fixture', revision: 'r1' });
    assert.equal(transformed.functions.length, 2);
    const modulePath = join(dir, 'fixture.mjs');
    writeFileSync(modulePath, transformed.code);
    const runtimeUrl = new URL('../scripts/code-corpus/capture-runtime.cjs', import.meta.url).href;
    const runner = `import ${JSON.stringify(runtimeUrl)}; const m = await import(${JSON.stringify(`file://${modulePath}`)}); const a={count:1}; m.mutate(a); const b={count:4}; try { m.fail(b); } catch {} `;
    const output = join(dir, 'capture.jsonl');
    const child = spawnSync(process.execPath, ['--input-type=module', '-e', runner], {
      encoding: 'utf8', env: { ...process.env, CODE_CORPUS_CAPTURE: output },
    });
    assert.equal(child.status, 0, child.stderr);
    const rows = readFileSync(output, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(rows.length, 2);
    assert.deepEqual(rows[0].args, [{ count: 1 }, 2]);
    assert.equal(rows[0].expected, 3);
    assert.deepEqual(rows[0].input_after, [{ count: 3 }, 2]);
    assert.equal(rows[0].outcome, 'return');
    assert.deepEqual(rows[1].thrown, { name: 'Error', message: 'bad input' });
    assert.deepEqual(rows[1].input_after, [{ count: 5 }]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('runtime snapshots explicitly mark unsupported values and aliasing', () => {
  const require = createRequire(import.meta.url);
  const { snapshot } = require('../scripts/code-corpus/capture-runtime.cjs');
  const basic = snapshot([undefined, NaN, () => 1]);
  assert.deepEqual(basic.value, [null, null, null]);
  assert.equal(basic.portable, false);
  assert.deepEqual(basic.reasons.map((r) => r.reason), ['undefined', 'NaN', 'function']);
  const shared = {};
  assert.deepEqual(snapshot([shared, shared]).reasons.map((r) => r.reason), ['cycle-or-alias']);
  assert.equal(snapshot({ unsupported: 'this is ordinary user data' }).portable, true);
  for (const [value, reason] of [[new Date(), 'Date'], [new Map(), 'Map'], [new Set(), 'Set'], [Object.create({ x: 1 }), 'non-plain prototype']]) {
    assert.equal(snapshot(value).reasons[0].reason, reason);
  }
  const sparse = []; sparse.length = 1;
  assert.equal(snapshot(sparse).reasons[0].reason, 'sparse array slot');
  let touched = false;
  const accessor = Object.defineProperty({}, 'x', { enumerable: true, get() { touched = true; return 1; } });
  assert.equal(snapshot(accessor).reasons[0].reason, 'accessor property');
  assert.equal(touched, false);
  const hostile = new Proxy({}, { ownKeys() { throw new Error('blocked'); } });
  assert.equal(snapshot(hostile).portable, false);
  assert.match(snapshot(hostile).reasons[0].reason, /snapshot failed/);
});

test('instrumentation skips reserved names and captures recursive and implicit undefined returns', () => {
  const dir = mkdtempSync(join(tmpdir(), 'code-corpus-recursive-'));
  try {
    const source = `export function recurse(n) { if (n <= 0) return 0; return recurse(n - 1) + 1; }\nexport function noReturn() {}\nexport function collision(__cc_args) { return __cc_args; }`;
    const transformed = instrumentSource(source, { path: 'recursive.js' });
    assert.deepEqual(transformed.functions.map((f) => f.instrumented), [true, true, false]);
    assert.match(transformed.functions[2].reason, /reserved instrumentation identifier/);
    const modulePath = join(dir, 'fixture.mjs'); writeFileSync(modulePath, transformed.code);
    const runtimeUrl = new URL('../scripts/code-corpus/capture-runtime.cjs', import.meta.url).href;
    const runner = `import ${JSON.stringify(runtimeUrl)}; const m = await import(${JSON.stringify(`file://${modulePath}`)}); m.recurse(2); m.noReturn();`;
    const output = join(dir, 'capture.jsonl');
    const child = spawnSync(process.execPath, ['--input-type=module', '-e', runner], { encoding: 'utf8', env: { ...process.env, CODE_CORPUS_CAPTURE: output } });
    assert.equal(child.status, 0, child.stderr);
    const rows = readFileSync(output, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(rows.length, 4);
    assert.deepEqual(rows.filter((row) => row.function === 'recurse').map((row) => row.expected), [0, 1, 2]);
    const empty = rows.find((row) => row.function === 'noReturn');
    assert.equal(empty.outcome, 'return');
    assert.equal(empty.portable, false);
    assert.equal(empty.reasons[0].reason, 'undefined');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
