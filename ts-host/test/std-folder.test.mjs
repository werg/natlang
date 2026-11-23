// @natlang/std workspace combinators (proposeAndVerify, branchAndSelect) over folder transactions, the library in
// in-memory (browser-style) programs and callable folders, and examples/release's publishing.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { compileVirtualProject, createNatlangRuntime, Folder, loadVirtualCallables, nodePlatform } from '../dist/index.js';
import { branchAndSelect, proposeAndVerify } from '../dist/std/index.js';
import { buildExample, localModel } from './support/std.mjs';

const write = (path, text) => async folder => { await folder.file(path).writeText(text); return text; };

test('proposeAndVerify accepts a proposal only when the verifier passes, re-proposing on the failed draft', async () => {
  const folder = Folder.fromFiles({ 'README.md': '# app\n' }, 'write');
  const calls = [];
  // The reducer appends one line per attempt; the verifier wants two lines.
  const reducer = async (draft, label, problems) => {
    calls.push({ label, problems });
    const file = draft.file('notes.md');
    const before = await file.exists() ? await file.readText() : '';
    await file.writeText(`${before}${label}\n`);
    return before.split('\n').filter(Boolean).length + 1;
  };
  const verify = async proposal => {
    const lines = (await proposal.folder.readText('notes.md')).split('\n').filter(Boolean);
    return lines.length === 2 ? [] : [`notes.md has ${lines.length} lines; it needs 2.`];
  };
  const traces = [];
  const result = await createNatlangRuntime({ trace: trace => traces.push(trace) }).run(() =>
    proposeAndVerify(folder, reducer, { args: ['line'], verify, attempts: 3 }));
  assert.equal(result.accepted, true); assert.equal(result.value, 2); assert.equal(result.attempts, 1);
  assert.deepEqual(calls, [{ label: 'line', problems: [] }, { label: 'line', problems: ['notes.md has 1 lines; it needs 2.'] }]);
  assert.equal(await folder.readText('notes.md'), 'line\nline\n', 'the second draft continued the first');
  assert.deepEqual(result.history, [['notes.md has 1 lines; it needs 2.'], []]);
  const span = traces.find(trace => trace.definitionId === 'std:proposeAndVerify');
  assert.deepEqual(span.events.filter(event => event.kind === 'std_stage').map(event => event.stage), ['propose', 'verify', 'accept']);
  assert.ok(span.events.find(event => event.stage === 'verify').call_ids.some(id => id.includes('std-semanticRepair')), 'the repair loop is nested by reference');
});

test('proposeAndVerify installs nothing when no proposal passes', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'a\n' }, 'write');
  const result = await createNatlangRuntime({}).run(() => proposeAndVerify(folder, write('b.txt', 'b\n'),
    { verify: () => ['never good enough'], attempts: 2 }));
  assert.equal(result.accepted, false); assert.deepEqual(result.problems, ['never good enough']);
  assert.equal(result.attempts, 1, 'the second draft repeats the first, which ends the repair');
  assert.equal(folder.isFile('b.txt'), false);
  assert.equal(await result.proposal.folder.readText('b.txt'), 'b\n', 'the closest proposal is returned for inspection');
  await assert.rejects(createNatlangRuntime({}).run(() => proposeAndVerify({}, write('x', 'y'), { verify: () => [], attempts: 1 })),
    /std-arguments: proposeAndVerify works on a Folder or FolderHandle/);
});

test('branchAndSelect: each branch on its own copy, failing ones dropped, the goal judged pairwise by beats', async () => {
  const folder = Folder.fromFiles({ 'README.md': '# app\n' }, 'write');
  const verify = async proposal => (await proposal.folder.readText('ANNOUNCE.md')).includes('TODO') ? ['The announcement has a TODO.'] : [];
  const local = localModel(() => null, (options, prompt) => /serves goal better/.test(prompt) && /Breaking/.test(prompt) ? [0.8, 0.2] : [0.3, 0.7]);
  const traces = [];
  const result = await createNatlangRuntime({ model: local.driver, trace: trace => traces.push(trace) }).run(() => branchAndSelect(folder, [
    write('ANNOUNCE.md', 'Uploads resume.\n'),
    { reducer: write('ANNOUNCE.md', 'TODO\n') },
    write('ANNOUNCE.md', 'Breaking: /v1 is gone. Uploads resume.\n'),
  ], { verify, goal: 'upgraders learn first what breaks' }));
  assert.deepEqual(result.candidates.map(candidate => candidate.problems.length), [0, 1, 0]);
  assert.equal(result.chosen, 2);
  assert.equal(await folder.readText('ANNOUNCE.md'), 'Breaking: /v1 is gone. Uploads resume.\n');
  assert.equal(local.decisions.filter(decision => /serves goal better/.test(decision.prompt)).length, 1, 'one pairwise judgment for two passing branches');
  const span = traces.find(trace => trace.definitionId === 'std:branchAndSelect');
  assert.deepEqual(span.events.filter(event => event.kind === 'std_stage').map(event => event.stage), ['propose', 'verify', 'select', 'accept']);
  // Without a goal the first passing branch is taken; select chooses instead when given; nothing passing installs nothing.
  const plain = Folder.fromFiles({}, 'write');
  assert.equal((await createNatlangRuntime({}).run(() => branchAndSelect(plain, [write('x', '1'), write('x', '2')]))).chosen, 0);
  const picked = Folder.fromFiles({}, 'write');
  assert.equal((await createNatlangRuntime({}).run(() => branchAndSelect(picked, [write('x', '1'), write('x', '2')],
    { select: candidates => candidates.at(-1).index }))).chosen, 1);
  assert.equal(await picked.readText('x'), '2');
  const none = Folder.fromFiles({}, 'write');
  const failed = await createNatlangRuntime({}).run(() => branchAndSelect(none, [write('x', '1')], { verify: () => ['no'] }));
  assert.equal(failed.chosen, null); assert.equal(none.isFile('x'), false);
});

test('an in-memory program imports @natlang/browser/std and @natlang/node/std: the same library instance', async () => {
  const runtime = createNatlangRuntime({});
  const project = compileVirtualProject({ files: {
    'main.ts': `import { semanticRepair } from '@natlang/browser/std';
import * as node from '@natlang/node/std';
export const same = semanticRepair === node.semanticRepair;
export async function run(): Promise<number> {
  const repaired = await semanticRepair(10, { check: (value: number) => value > 3 ? ['too big'] : [], fix: (value: number) => value - 4, attempts: 5 });
  return repaired.value;
}
` } }, await import('../dist/index.js'), { platform: nodePlatform() });
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const main = project.require('main.ts');
  assert.equal(main.same, true);
  assert.equal(await runtime.run(() => main.run()), 2);
});

test('callable-folder TypeScript imports @natlang/node/std', async () => {
  const tree = loadVirtualCallables({ 'shorten.ts': `import { semanticRepair } from '@natlang/node/std';
export default async function shorten(words: string[]): Promise<string[]> {
  const result = await semanticRepair(words, { check: (value: string[]) => value.length > 2 ? ['too long'] : [], fix: (value: string[]) => value.slice(1), attempts: 3 });
  return result.value;
}
` }, '', '/project', nodePlatform());
  assert.deepEqual(await createNatlangRuntime({}).run(() => tree.shorten(['a', 'b', 'c', 'd'])), ['c', 'd']);
});

test('the browser bundle ships std.js beside natlang.js, sharing its chunks', () => {
  const dist = fileURLToPath(new URL('../dist/browser/', import.meta.url));
  if (!existsSync(`${dist}natlang.js`)) return;
  const std = readFileSync(`${dist}std.js`, 'utf8');
  const chunks = [...std.matchAll(/from ?"\.\/(chunks\/[^"]+)"/g)].map(match => match[1]);
  assert.ok(chunks.length, 'std.js imports the shared runtime chunks');
  const natlang = readFileSync(`${dist}natlang.js`, 'utf8');
  assert.ok(chunks.some(chunk => natlang.includes(chunk)), 'natlang.js imports the same chunk');
});

test('examples/release publish: the release section through proposeAndVerify, the announcement through branchAndSelect', async () => {
  const load = await buildExample('release');
  const { publish, announce, sectionProblems } = await load('publish.js');
  const notes = ['#412 Uploads resume after a dropped connection.', '#418 /v1/sync is removed; use /v2/sync.'];
  let sections = 0;
  const local = localModel(opening => {
    // The first section forgets the heading; the retry, shown its problems, writes it.
    if (opening.includes('Write the release section')) return ++sections === 1 ? 'return notes.map(note => "- " + note).join("\\n")' :
      'return "## " + version + "\\n" + notes.map(note => "- " + note).join("\\n")';
    if (opening.includes('release announcement')) return 'return style.startsWith("plain") ? "Version 3.0 removes /v1/sync: move to /v2/sync. Uploads also resume." : notes.map(note => "* " + note).join("\\n")';
    return null;
  }, (options, prompt) => /serves goal better/.test(prompt) ? [0.3, 0.7] : [0.5, 0.5]);
  const folder = Folder.fromFiles({ 'README.md': '# Sync\n' }, 'write');
  const runtime = createNatlangRuntime({ model: local.driver });
  const section = await runtime.run(() => publish(folder, '3.0.0', notes));
  assert.equal(section.accepted, true); assert.equal(section.attempts, 1);
  assert.match(section.history[0][0], /not "## 3\.0\.0"/);
  assert.match(await folder.readText('releases/3.0.0.md'), /^## 3\.0\.0\n- #412/);
  assert.ok(local.openings.some(opening => opening.includes('The first line is')), 'the retry saw the verifier problem');
  const announcement = await runtime.run(() => announce(folder, notes));
  assert.equal(announcement.chosen, 0, 'beats kept the plain announcement that leads with the breaking change');
  assert.match(await folder.readText('ANNOUNCEMENT.md'), /removes \/v1\/sync/);
  assert.equal(typeof sectionProblems, 'function');
});
