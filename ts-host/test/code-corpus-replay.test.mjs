import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { replayIsolated, project, materializeCorpus } from '../scripts/code-corpus/replay.mjs';
import { readJsonl, writeJsonl, digest } from '../scripts/code-corpus/common.mjs';
import { extractFunctions } from '../scripts/code-corpus/extract.mjs';
const task = () => ({version:'natlang.code_task/1',id:'fixture:double',group_id:'fixture:double',kind:'function',instruction:'Return twice the input.',source:{name:'fixture',license:'MIT'},function:{name:'f',parameters:[{name:'x'}],body:'{ return x * 2; }'},cases:[{args:[3],expected:6,outcome:'return'},{args:[4],expected:8,outcome:'return'}]});
test('replay produces a real eval turn and a done turn with source-level split groups', async () => {
  const a = task();
  const row = await replayIsolated(a,0);
  assert.equal(row.outcome.accepted,true);
  assert.deepEqual(row.outcome.action_ledger.map(e=>e.name),['eval']);
  const result = await materializeCorpus([row]);
  assert.equal(result.turns.length,2);
  assert.deepEqual(result.turns[0].source_groups,['fixture:double']);
  assert.equal(result.turns[0].license,'MIT');
  assert.equal(project(a,0).program.split,project(a,1).program.split);
  assert.match(project(a).program.semantics.files[project(a).program.semantics.root],/args:\n  x: "number"\nreturns: "number"/);
});
test('wrong outputs are not admitted, mutation and nonportable shapes are rejected', async () => {
  const a = task(); a.cases[0].expected=7;
  const row=await replayIsolated(a,0); assert.equal(row.outcome.accepted,false);
  assert.equal((await materializeCorpus([row])).turns.length,0);
  a.cases[0].input_after=[99]; assert.throws(()=>project(a),/mutating/);
  const b=task(); b.cases[0].args=[{}]; assert.throws(()=>project(b),/portable replay/);
  const c=task(); c.cases[1].args=[3]; assert.throws(()=>project(c),/conflicting outputs/);
});
test('runaway code is terminated', async () => {
  const a=task(); a.function.body='{ for (let i = 0; i < 1e15; i++) {} return x; }';
  await assert.rejects(replayIsolated(a,0,500),/timeout/);
});
test('nested arrays replay using native list type syntax', async () => {
  const a=task(); a.function.body='{ return [x]; }';
  a.cases=[{args:[[1,2]],expected:[[1,2]],outcome:'return'}];
  assert.equal((await replayIsolated(a,0)).outcome.accepted,true);
});
test('empty-only captured arrays can use explicit primitive array boundary types', async () => {
  const a=task(); a.function.parameters=[{name:'x',type:'number[]'}];
  a.function.return_type='number[]'; a.function.body='{ return x; }';
  a.cases=[{args:[[]],expected:[],outcome:'return'}];
  assert.equal((await replayIsolated(a,0)).outcome.accepted,true);
  a.function.parameters[0].type='Unresolved[]';
  assert.throws(()=>project(a),/empty-only array/);
});
test('extracted sibling function closure replays without module initializers', async () => {
  const [record] = extractFunctions('/** Double the input. */ export function twice(x: number): number { return helper(x); }\nexport function helper(x: number): number { return add(x, x); }\nfunction add(a: number, b: number): number { return a + b; }\nconst secretState = launchExternalEffect();', {path:'fixture.ts',sourceName:'fixture'});
  record.cases = task().cases;
  assert.equal(record.function.helpers.length, 2);
  assert.doesNotMatch(record.function.helpers.join('\n'), /export|secretState|launchExternalEffect/);
  const projected = project(record, 0);
  assert.deepEqual(Object.keys(projected.program.semantics.files).sort(), ['twice.nl', 'twice/helper.ts', 'twice/helper/add.ts']);
  assert.equal(projected.program.source_layout.subfunctions.add, 'twice/helper/add.ts');
  const row = await replayIsolated(record, 0);
  assert.equal(row.outcome.accepted, true);
  assert.doesNotMatch(row.trajectory[0].assistant.calls[0].arguments.code, /function helper/);
});
test('recursive and untyped subfunction graphs are excluded from natlang replay', () => {
  const direct = task(); direct.function.body = '{ return f(x); }';
  assert.throws(() => project(direct), /Recursive function call/);
  const indirect = task(); indirect.function.body = '{ return helper(x); }';
  indirect.function.helpers = ['function helper(x: number): number { return f(x); }'];
  assert.throws(() => project(indirect), /Recursive subfunction call graph/);
  const untyped = task(); untyped.function.body = '{ return helper(x); }';
  untyped.function.helpers = ['function helper(x) { return x * 2; }'];
  assert.throws(() => project(untyped), /needs explicit portable parameter and return types/);
});
test('dependency-bearing corpus replay resolves captured relative imports from the application workspace', async t => {
  const root = await mkdtemp(join(tmpdir(), 'corpus-replay-workspace-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(join(root, 'package.json'), JSON.stringify({ name: 'corpus-replay-fixture', private: true, type: 'module' }));
  await mkdir(join(root, 'src'));
  await writeFile(join(root, 'src', 'helper.ts'), 'export default function double(value: number): number { return value * 2; }\n');
  const record = task();
  record.source.path = 'src/main.ts';
  record.function.imports = [{ specifier: './helper.ts', source: "import double from './helper.ts';" }];
  record.function.body = '{ return double(x); }';
  await assert.rejects(replayIsolated(record, 0), /Local subfunction conversion requires the source workspace/);
  const row = await replayIsolated(record, 0, 10000, { workspace: root });
  assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome));
  assert.equal(row.outcome.value, 6);
  assert.deepEqual(Object.keys(row.task.program_ir.semantics.files).sort(), ['f.nl', 'f/double.ts']);
  assert.doesNotMatch(row.trajectory[0].assistant.calls[0].arguments.code, /import double/);
});
test('dependency-bearing corpus replay imports installed local packages from the workspace', async t => {
  const root = await mkdtemp(join(tmpdir(), 'corpus-replay-package-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(join(root, 'package.json'), JSON.stringify({ name: 'corpus-replay-package-fixture', private: true, type: 'module' }));
  const installed = join(root, 'node_modules', 'corpus-replay-local');
  await mkdir(installed, { recursive: true });
  await writeFile(join(installed, 'package.json'), JSON.stringify({ name: 'corpus-replay-local', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(installed, 'index.js'), 'export class LocalBox { constructor(value) { this.value = value; } triple() { return this.value * 3; } }\n');
  const record = task();
  record.source.path = 'src/main.ts';
  record.function.imports = [{ specifier: 'corpus-replay-local', source: "import { LocalBox } from 'corpus-replay-local';" }];
  record.function.body = '{ const box = new LocalBox(x); return box.triple(); }';
  record.cases = [{ args: [3], expected: 9, outcome: 'return' }, { args: [4], expected: 12, outcome: 'return' }];
  const row = await replayIsolated(record, 0, 10000, { workspace: root });
  assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome));
  assert.equal(row.outcome.value, 9);
  assert.equal(row.provenance.workspace_before.path, root);
  assert.ok(row.provenance.workspace_before.hashes['package.json']);
  assert.ok(row.outcome.effects.host_events.some(event => event.operation === 'packages.import' && event.specifier === 'corpus-replay-local'));
});
test('JSONL is bounded, atomic and refuses replacement', async () => {
  const dir=await mkdtemp(join(tmpdir(),'corpus-common-'));
  try { const path=join(dir,'rows.jsonl'); await writeJsonl(path,[{a:1},{a:2}]);
    assert.deepEqual(await readJsonl(path,{limit:1}),[{a:1}]);
    await assert.rejects(writeJsonl(path,[{a:3}]),/EEXIST/);
    assert.equal((await readJsonl(path)).length,2);
    assert.equal(digest({b:2,a:1}),digest({a:1,b:2}));
  } finally { await rm(dir,{recursive:true,force:true}); }
});
