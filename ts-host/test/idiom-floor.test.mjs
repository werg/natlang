import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { checkFixedCursorIterations, checkNlIdioms, IDIOM_CODES } from '../dist/compiler/idiom.js';
import { checkProject } from '../dist/compiler/node-project.js';
import { compileProject } from '../dist/compiler/project.js';
import { virtualProjectFiles } from '../dist/runtime/virtual-project.js';
import { naturalExamples } from '../playground/examples/natural.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const idioms = diagnostics => diagnostics.filter(item => IDIOM_CODES.includes(item.code));

/** A virtual project's idiom warnings. */
const floor = files => idioms(compileProject({ project: '/p', files: virtualProjectFiles(files, '/p'), emit: false, write: false,
  module: 'commonjs', surfaceSpecifiers: ['@natlang/node'] }).diagnostics);

const iterate = body => checkFixedCursorIterations(ts.createSourceFile('a.ts',
  `import { iterateOn, nl } from '@natlang/node';\nimport judge from './judge.nl';\n${body}`, ts.ScriptTarget.ES2022, true));

test('a leaf .nl with a finite result and no readout gets the decision-readout warning (playground feedback-tone)', () => {
  const tone = naturalExamples.find(item => item.id === 'feedback-tone');
  const found = floor({ 'natlang.json': '{}', ...tone.files });
  assert.deepEqual(found.map(item => [item.code, item.file, item.line, item.severity]),
    [['idiom-decision-readout', 'feedback/tone.nl', 4, 'warning']]);
  assert.equal(found[0].message, '`tone.nl` returns one of 3 values: add `readout: decision` so the model reasons and scores them, ' +
    'or keep the tool loop if it must call helpers first.');
  // boolean is finite too; a declared readout, a companion folder, uses:, a Folder parameter or a non-finite result is not flagged.
  assert.equal(floor({ 'natlang.json': '{}', 'ok.nl': '---\nargs:\n  x: string\nreturns: boolean\n---\nJudge x.\n' })[0]?.code, 'idiom-decision-readout');
  for (const files of [
    { 'a.nl': '---\nargs:\n  x: string\nreturns: boolean\nreadout: decision\n---\nJudge x.\n' },
    { 'a.nl': '---\nargs:\n  x: string\nreturns: boolean\n---\nJudge x with norm.\n', 'a/norm.ts': 'export default function norm(x: string): string { return x.trim(); }\n' },
    { 'a.nl': '---\nargs:\n  x: Folder\nreturns: boolean\n---\nJudge the folder x.\n' },
    { 'a.nl': '---\nargs:\n  x: string\nreturns: string\n---\nSummarize x.\n' },
    { 'a.nl': '---\nargs:\n  x: string\nreturns: \'"only"\'\n---\nSay only.\n' },
  ]) assert.deepEqual(floor({ 'natlang.json': '{}', ...files }), [], JSON.stringify(files));
});

test('code tokens in a .nl body get the code-in-prose warning (cave-mission readFrames), fenced examples do not', () => {
  const path = 'applications/cave-mission/planMission/survey/readFrames.nl';
  const text = readFileSync(join(repo, path), 'utf8');
  const record = { kind: 'natlang', subtype: 'function', name: 'readFrames', source: path, text, args: {}, returns: 'string', types: {},
    codebase: { advanceFrameUnit: {} } };
  const found = checkNlIdioms(record, path);
  assert.deepEqual(found.map(item => [item.code, item.line]), [['idiom-code-in-prose', 30]]);
  assert.equal(found[0].message, '`readFrames.nl` describes code (`=>`) in its instructions: write that step as a TypeScript helper in ' +
    '`applications/cave-mission/planMission/survey/readFrames/` and call it from the function.');
  for (const [body, token] of [['Read JSON.parse(x) first.', 'JSON.parse('], ['Take items.slice(1).', '.slice('],
    ['Use Math.max of the scores.', 'Math.'], ['Answer with eval({code:"return 1"}).', 'eval({']]) {
    const hit = floor({ 'natlang.json': '{}', 'f.nl': `---\nargs:\n  x: string\nreturns: string\n---\n${body}\n` });
    assert.equal(hit.length, 1, body); assert.match(hit[0].message, new RegExp(token.replace(/[.(){}]/g, '\\$&')));
  }
  assert.deepEqual(floor({ 'natlang.json': '{}', 'f.nl': '---\nargs:\n  x: string\nreturns: string\n---\nSummarize x.\n\n```ts\nconst y = x.slice(1);\n```\n' }), []);
  // One warning per file, at the first token.
  assert.equal(floor({ 'natlang.json': '{}', 'f.nl': '---\nargs:\n  x: string\nreturns: string\n---\nUse a => b.\nThen Math.max.\n' }).length, 1);
});

test('an iterateOn that walks a fixed list one item at a time gets the fixed-cursor warning', () => {
  for (const body of [
    'type S = { i: number; out: string[] };\nconst step = async (s: S): Promise<S> => ({ i: s.i + 1, out: [...s.out, await judge(files[s.i])] });\n' +
      'const done = await iterateOn(step, { i: 0, out: [] } as S).withLimit({ maxSteps: files.length }).until(s => s.i >= files.length);',
    'const done = await iterateOn(async (s: { cursor: number; answers: Record<string, string> }) => {\n  const file = task.files[s.cursor];\n' +
      '  const answer = await judge(file);\n  return { cursor: s.cursor + 1, answers: { ...s.answers, [file]: answer } };\n}, { cursor: 0, answers: {} }).until(s => s.cursor >= task.files.length);',
    'const step = async ({ queue, hits }: { queue: string[]; hits: number }) => ({ queue: queue.slice(1), hits: hits + (await judge(queue[0]) ? 1 : 0) });\n' +
      'const done = await step.iterateOn({ queue: items, hits: 0 }).withMeasure(s => s.queue.length).until(s => s.queue.length === 0);',
  ]) {
    const found = iterate(body);
    assert.equal(found.length, 1, body);
    assert.equal(found[0].code, 'idiom-fixed-cursor-iterate');
    assert.equal(found[0].severity, 'warning');
    assert.equal(found[0].message, 'This `iterateOn` steps through a fixed list one item at a time: map the step over the list with ' +
      '`Promise.all`, and keep `iterateOn` for steps that depend on the previous state.');
  }
});

test('iterations whose steps depend on the previous state, stop early or change the list are left alone', () => {
  for (const body of [
    // The judgment reads what earlier steps produced.
    'const step = async (s: { i: number; notes: string }) => ({ i: s.i + 1, notes: await judge(files[s.i], s.notes) });\n' +
      'await iterateOn(step, { i: 0, notes: "" }).until(s => s.i >= files.length);',
    // Stops on a result.
    'const step = async (s: { i: number; found: boolean }) => ({ i: s.i + 1, found: await judge(files[s.i]) });\n' +
      'await iterateOn(step, { i: 0, found: false }).until(s => s.found || s.i >= files.length);',
    // The queue grows.
    'const step = async (s: { queue: string[] }) => ({ queue: [...s.queue.slice(1), ...(await judge(s.queue[0]))] });\n' +
      'await iterateOn(step, { queue: [root] }).until(s => s.queue.length === 0);',
    // The whole state goes to the model.
    'const step = async (s: { i: number; draft: string }) => ({ i: s.i + 1, draft: await judge(s) });\n' +
      'await iterateOn(step, { i: 0, draft: "" }).until(s => s.i >= passes.length);',
    // A natural-language step or predicate.
    'await judge.iterateOn({ i: 0 }).until(s => s.i >= 3);',
    'const step = async (s: { i: number }) => ({ i: s.i + 1 });\nawait iterateOn(step, { i: 0 }).until(nl<(s: { i: number }) => boolean>`s is done.`);',
    // An effect on outer state.
    'let total = 0;\nconst step = async (s: { i: number }) => { total += await judge(files[s.i]); return { i: s.i + 1 }; };\n' +
      'await iterateOn(step, { i: 0 }).until(s => s.i >= files.length);',
    // Reads other elements of the list.
    'const step = async (s: { i: number; out: string[] }) => ({ i: s.i + 1, out: [...s.out, await judge(files[s.i], files[s.i - 1])] });\n' +
      'await iterateOn(step, { i: 0, out: [] }).until(s => s.i >= files.length);',
    // Not a cursor: an open-ended search.
    'const step = async (s: { text: string }) => ({ text: await judge(s.text) });\nawait iterateOn(step, { text: draft }).withLimit({ maxSteps: 3 }).until(s => s.text.length < 80);',
  ]) assert.deepEqual(iterate(body), [], body);
});

test('the curated programs raise no idiom-floor warning, except the one finite helpdesk judgment left to WS7', () => {
  const sites = dir => idioms(checkProject(join(repo, dir), { write: false }).diagnostics).map(item => `${item.file}:${item.line} ${item.code}`);
  for (const dir of ['examples/triage', 'examples/review', 'applications/notebook']) assert.deepEqual(sites(dir), [], dir);
  // triage/urgency.nl is a leaf with a finite result and no readout: the rule holds there, and the program change
  // (readout: decision, or helpers in a companion folder) belongs to the migration, not to the check.
  assert.deepEqual(sites('applications/helpdesk'), ['triage/urgency.nl:5 idiom-decision-readout']);
});
