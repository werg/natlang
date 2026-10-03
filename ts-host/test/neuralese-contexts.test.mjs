import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, Context, ContextError, createNatlangRuntime, defineNatlang, live, loadNatlang, nodeSourceFiles,
  nzExports, save, softFunction, loadNz, encodeNz, decodeNz, NzFileError, declareDistribution, distributionOf } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { inline } from '../dist/runtime/lowered.js';
import { callableMeta } from '../dist/runtime/callable.js';

const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

const standIn = (width = 8) => {
  const store = new MemoryNeuraleseStore();
  return { store, port: new StandInNeuralesePort(store, hashingEmbedder(width), width, 'nd:natlang@1') };
};
const neuraleseDriver = fn => Object.assign(fn, { neuralese: true });
/** Every Neuralese part and all text a request carried. */
const requestParts = messages => messages.flatMap(message => [message.content, ...(message.tool_calls ?? []).map(call => call.function.arguments)])
  .flatMap(content => Array.isArray(content) ? content : [{ type: 'text', text: String(content ?? '') }]);

/** A temporary folder with the given files (relative path → text or bytes). */
function folder(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-ctx-'));
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), content);
  }
  return root;
}
const withBytes = root => ({ ...nodeSourceFiles(root), readBytes: path => new Uint8Array(readFileSync(path)) });

const SCORER = `---
args:
  ticket: string
returns: number
---
Score the ticket with score.
`;
const scoreModule = argType => `export default async function score(x: ${argType}): Promise<number> { return 1; }\n`;

async function rubricFile(store, port) {
  const rubric = await port.write('prefer email follow ups');
  const body = await port.write('label the ticket using the rubric');
  const triage = softFunction({ type: '(t: string) => string', body: body.id,
    captures: { rubric: neuraleseRef('Neuralese<Rubric>', rubric.id) } });
  const bytes = await save({ rubric: { type: 'Neuralese<Rubric>', value: neuraleseRef('Neuralese<Rubric>', rubric.id) },
    triage, limits: { type: '{ maxItems: number, note: Neuralese<string> }',
      value: { maxItems: 20, note: neuraleseRef('Neuralese<string>', rubric.id) } } },
  { store, dialect: 'nd:natlang@1', types: 'type Rubric = { criteria: string[] };' });
  return { bytes, rubric, body };
}

test('.nz files round-trip typed exports, soft functions with snapshot captures, and distributional blocks', async () => {
  const { store, port } = standIn();
  const { bytes, rubric, body } = await rubricFile(store, port);
  assert.deepEqual([...bytes.subarray(0, 8)].slice(4), [0, 0, 0, 0], 'safetensors header length is a little-endian u64');
  const fresh = new MemoryNeuraleseStore();
  const loaded = await loadNz(bytes, { store: fresh });
  assert.deepEqual(Object.keys(loaded.exports).sort(), ['limits', 'rubric', 'triage']);
  assert.equal(loaded.exports.rubric.$neuralese.id, rubric.id);
  assert.equal(loaded.exports.triage.kind, 'soft-function');
  assert.equal(loaded.exports.triage.body, body.id);
  assert.equal(loaded.exports.triage.captures.rubric.$neuralese.id, rubric.id);
  assert.equal(loaded.exports.limits.maxItems, 20);
  assert.ok(await fresh.has(rubric.id) && await fresh.has(body.id), 'loading puts blocks into the store');
  // Equal content gives equal bytes.
  const again = await rubricFile(store, port);
  assert.deepEqual(again.bytes, bytes);
  // A Gaussian block keeps its log-scale tensor through save and load.
  const logScale = await port.write('small scale');
  const mean = await port.write('mean vectors');
  const meanMeta = await store.meta(mean.id), scaleMeta = await store.meta(logScale.id);
  assert.equal(meanMeta.length, 2); assert.equal(scaleMeta.length, 2);
  declareDistribution(mean.id, logScale.id);
  const gaussian = await save({ belief: neuraleseRef('Neuralese<string>', mean.id) }, { store, dialect: 'nd:natlang@1' });
  const header = decodeNz(gaussian).header;
  assert.equal(header.exports.belief.value.$neuralese.distribution, 'gaussian-diag');
  assert.equal(header.exports.belief.value.$neuralese.logScale, logScale.id);
  await loadNz(gaussian);
  assert.equal(distributionOf(mean.id).logScale, logScale.id);
});

test('loading checks types, dialects, block integrity and reference cycles (nz-load-and-typecheck, nz-block-integrity)', async () => {
  const { store, port } = standIn();
  const block = await port.write('note text');
  const data = (await store.get(block.id)).data;
  const header = exports => ({ format: 'natlang.neuralese-file/1', dialect: 'nd:natlang@1', exports,
    blocks: { [block.id]: { dialect: 'nd:natlang@1', length: block.length, width: block.width, dtype: 'F32' } } });
  const encode = exports => encodeNz(header(exports), new Map([[block.id, data]]));
  const note = { $neuralese: { type: 'Neuralese<string>', id: block.id } };
  await assert.rejects(() => loadNz(encode({ limits: { type: '{ maxItems: number, note: Neuralese<string> }',
    value: { maxItems: 'twenty', note } } })), error => error instanceof NzFileError && error.code === 'neuralese-file-type' &&
    /export limits/.test(error.message));
  await assert.rejects(() => loadNz(encode({ note: { type: 'Neuralese<string, "nd:other@2">', value: note } })),
    error => error.code === 'neuralese-dialect-mismatch');
  await assert.rejects(() => loadNz(encode({ a: { type: 'string', value: { $ref: 'b' } }, b: { type: 'string', value: { $ref: 'a' } } })),
    error => error.code === 'neuralese-ref-cycle');
  const good = encode({ note: { type: 'Neuralese<string>', value: note } });
  await loadNz(good);
  const tampered = good.slice();
  tampered[tampered.length - 1] ^= 0xff;
  await assert.rejects(() => loadNz(tampered), error => error.code === 'neuralese-block-integrity');
});

test('a soft function with live captures cannot be saved; a snapshot capture keeps its value (nz-live-capture-save, fn-snapshot-capture)', async () => {
  const { store, port } = standIn();
  const body = await port.write('increment the counter');
  let count = 0;
  const counting = softFunction({ type: '() => number', body: body.id, captures: { count: live(() => count, value => { count = value; }) } });
  await assert.rejects(() => save({ counting }, { store, dialect: 'nd:natlang@1' }), error => error.code === 'neuralese-live-capture-save');
  let x = 1;
  const seen = [];
  const snap = softFunction({ type: '() => number', body: body.id, captures: { x } });
  x = 2;
  const driver = neuraleseDriver(({ messages }) => { seen.push(JSON.stringify(messages)); return { calls: [['return_result', { status: 'success', value: 1 }]] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  assert.equal(await runtime.run(() => snap()), 1);
  assert.match(seen[0], /\bx\b[^\n]*= 1\b/, 'the opening lists the snapshot value');
  assert.doesNotMatch(seen[0], /\bx\b[^\n]*= 2\b/);
});

test('live(x) captures are read per call and written back (fn-live-capture)', async () => {
  const { store, port } = standIn();
  const body = await port.write('increment n');
  let n = 0;
  const bump = softFunction({ type: '() => string', body: body.id, captures: { n: live(() => n, value => { n = value; }) } });
  let turn = 0;
  const driver = neuraleseDriver(() => turn++ % 2 === 0 ?
    { calls: [['eval', { code: 'n = n + 1;' }]] } : { calls: [['return_result', { status: 'success', value: 'ok' }]] });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  assert.equal(await runtime.run(() => bump()), 'ok');
  assert.equal(n, 1);
  assert.equal(await runtime.run(() => bump()), 'ok');
  assert.equal(n, 2);
});

test('a stored soft function shows its body and snapshot captures as blocks (nz-capture-resolution)', async () => {
  const { store, port } = standIn();
  const { bytes, rubric, body } = await rubricFile(store, port);
  const { triage } = nzExports(await loadNz(bytes, { store }));
  const requests = [];
  const driver = neuraleseDriver(({ messages }) => { requests.push(messages); return { calls: [['return_result', { status: 'success', value: 'billing' }]] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  assert.equal(await runtime.run(() => triage('I was charged twice')), 'billing');
  const ids = requestParts(requests[0]).filter(part => part.type === 'neuralese').map(part => part.id);
  assert.ok(ids.includes(body.id), 'the soft body is a block');
  assert.ok(ids.includes(rubric.id), 'the captured rubric is a block');
  assert.ok(requestParts(requests[0]).some(part => part.type === 'text' && /\brubric\b/.test(part.text)));
});

test('function-typed bindings are captured by value even in text nl (fn-function-capture-by-value)', () => {
  const target = (text, extra = {}) => ({ text, natlang: text, aliases: {}, ...extra });
  const plan = { definitionId: 'nl:test#1', sourceSpan: { file: 'app.ts', line: 1, start: 0, end: 1 }, strings: ['Call g.'],
    instructions: 'Call g.', parameters: [], returns: target('string'), inheritedCodebaseRevision: '',
    captures: [{ name: 'g', type: target('() => string', { host: { kind: 'function' } }), mutable: true },
      { name: 'count', type: target('number'), mutable: true }] };
  let g = () => 'hi', count = 1;
  const f = inline(plan, [], { g: [() => g, value => { g = value; }], count: [() => count, value => { count = value; }] });
  g = f; count = 2;
  const cells = callableMeta(f).captures;
  assert.notEqual(cells.g.get(), f, 'f cannot reach itself through g');
  assert.equal(cells.g.get()(), 'hi');
  assert.equal(cells.g.mutable, false);
  assert.equal(cells.count.get(), 2, 'data lets stay live');
  assert.equal(cells.count.mutable, true);
});

test('contexts are content addressed; one cannot contain itself (ctx-self-containment)', () => {
  const root = folder({ 'notes.md': 'hello' });
  const files = withBytes(root);
  return Context.fromFolder(root, files).then(ctx => {
    const outer = ctx.with({ self: ctx });
    assert.notEqual(outer.id, ctx.id);
    assert.ok(outer.contains(ctx));
    assert.ok(!outer.contains(outer), 'a changed context has a new ID, never a cycle');
    assert.equal(ctx.with({}).id, ctx.id, 'equal content, equal ID');
    assert.equal(ctx.data['notes.md'], 'hello');
  });
});

const SQL_SKILL = `---
name: sql
description: Joining tables in SQL queries.
natlang:
  scope:
    joinLimit: { type: number, value: 3, description: most tables one query may join }
---
Use explicit join keys.
`;

test('rebinding checks the context interface; adding data is free; executable nodes come from files', async () => {
  const root = folder({ 'foo.nl': SCORER, 'foo/score.ts': scoreModule('string'),
    'other/score.ts': scoreModule('number'), 'same/score.ts': scoreModule('string'),
    'same/notes.md': 'library notes', 'library/skills/sql/SKILL.md': SQL_SKILL, 'library/skills/web/SKILL.md': '---\nname: web\ndescription: Citing web sources.\n---\nCite sources.\n' });
  const foo = loadNatlang(join(root, 'foo.nl'), root);
  const files = withBytes(root);
  const wrong = await Context.fromFolder(join(root, 'other'), files);
  assert.throws(() => foo.in(wrong), error => error instanceof ContextError && error.code === 'context-interface-mismatch',
    'ctx-rebind-interface');
  const right = await Context.fromFolder(join(root, 'same'), files);
  const rebound = foo.in(right);
  assert.equal(typeof rebound.score, 'function');
  // ctx-no-new-executable: an nl function written at run time is not a file node.
  const extra = defineNatlang('---\nreturns: string\n---\nSay hi.\n', { name: 'extra' });
  assert.throws(() => right.with({ extra }), error => error.code === 'context-new-executable');
  // ctx-add-data and ctx-union: data entries and a picked library skill reach the call's scope.
  const { store, port } = standIn();
  const memo = await port.write('the customer prefers email');
  const library = await Context.fromFolder(join(root, 'library'), files);
  const program = Context.ofCallable(foo).union(library.pick('skills/sql'))
    .with({ 'skills/sql/notes.md': 'join on customer_id', memo: neuraleseRef('Neuralese<string>', memo.id) });
  assert.ok(!program.names.includes('skills/web/SKILL.md'));
  // Bound skills are disclosed progressively (S2): listed in the opening, their scope bindings injected, their
  // instructions and files read on request.
  const seen = [];
  const turns = [[['read_code', { name: 'skills.sql' }]], [['read_code', { name: 'skills.sql/notes.md' }]],
    [['return_result', { status: 'success', value: 3 }]]];
  const driver = neuraleseDriver(({ messages }) => { seen.push(messages); return { calls: turns[seen.length - 1] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  assert.equal(await runtime.run(() => foo.in(program)('broken export')), 3);
  const opening = JSON.stringify(seen[0]);
  assert.match(opening, /- sql: Joining tables in SQL queries\./);
  assert.match(opening, /const joinLimit: number = 3;  \/\/ from skill sql/);
  assert.doesNotMatch(opening, /Use explicit join keys|join on customer_id/, 'instructions wait until the model reads them');
  assert.match(JSON.stringify(seen[1].at(-1)), /Use explicit join keys.*- notes\.md/);
  assert.match(JSON.stringify(seen[2].at(-1)), /join on customer_id/);
  assert.ok(requestParts(seen[0]).some(part => part.type === 'neuralese' && part.id === memo.id));
});

test('an edited executable node must keep its signature (ctx-edit-executable)', async () => {
  const root = folder({ 'foo.nl': SCORER, 'foo/score.ts': scoreModule('string') });
  const foo = loadNatlang(join(root, 'foo.nl'), root);
  const ctx = Context.ofCallable(foo);
  assert.throws(() => ctx.edit('score', 'export default async function score(x: string): Promise<string> { return "1"; }\n'),
    error => error.code === 'context-interface-mismatch');
  const edited = ctx.edit('score', 'export default async function score(x: string): Promise<number> { return x.length; }\n');
  assert.notEqual(edited.id, ctx.id);
  assert.equal(typeof foo.in(edited).score, 'function');
});

test('.nz files in a folder are context data, and a project imports them with generated declarations (nz-import-declarations)', async () => {
  const { store, port } = standIn();
  const { bytes } = await rubricFile(store, port);
  const root = folder({
    'data/support.nz': bytes,
    'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true,
      allowArbitraryExtensions: true, skipLibCheck: true }, include: ['*.ts'] }),
    'package.json': JSON.stringify({ type: 'module' }),
    'app.ts': "import { rubric, triage, limits } from './data/support.nz';\nexport const items: number = limits.maxItems;\n" +
      "export const kept: Neuralese<{ criteria: string[] }> = rubric;\nexport const call: (t: string) => Promise<string> = triage;\n",
    'bad.ts': "import { rubric } from './data/support.nz';\nexport const peek = rubric.criteria;\n",
  });
  const ctx = await Context.fromFolder(join(root, 'data'), withBytes(root), { store });
  assert.equal(typeof ctx.data.support.triage, 'function');
  assert.equal(ctx.data.support.limits.maxItems, 20);
  const outDir = mkdtempSync(join(tmpdir(), 'natlang-nz-out-'));
  const built = buildProject({ project: root, outDir, runtimeModule });
  const declaration = built.declarations[join(root, 'data', 'support.d.nz.ts')];
  assert.match(declaration, /export declare const rubric: Neuralese<Rubric>;/);
  assert.match(declaration, /export declare const triage: Neuralese<\(t: string\) => string>;/);
  const errors = built.diagnostics.filter(item => item.severity === 'error');
  assert.ok(errors.length > 0 && errors.every(item => item.file === 'bad.ts'), JSON.stringify(errors));
  // Without the misuse, the project builds and its emitted module loads the file.
  writeFileSync(join(root, 'bad.ts'), 'export {};\n');
  const clean = buildProject({ project: root, outDir, runtimeModule });
  assert.equal(clean.ok, true, JSON.stringify(clean.diagnostics));
  const app = await import(pathToFileURL(join(outDir, 'app.js')).href);
  assert.equal(app.items, 20);
  assert.equal(typeof app.call, 'function');
  assert.match(app.kept.$neuralese.id, /^nz1_/);
});
