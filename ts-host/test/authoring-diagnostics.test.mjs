// Errors an author meets while building a program: each names the place and the fix (friction logs of the program
// inventor's calibration builders, 2026-11-25).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as natlang from '../dist/index.js';
import { replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const check = files => natlang.compileVirtualProject({ root: '/project', files }, natlang, {});
const ROOT = '---\nargs:\n  words: string[]\nreturns: number\n---\nCall find(words) and return its result.\n';

test('a refused loop in callable-folder TypeScript is reported at the loop, with the counted form to write', () => {
  const compiled = check({ 'root.nl': ROOT, 'root/find.ts': 'export default function find(tokens: string[]): number {\n' +
    '  const run = [\'a\', \'b\'];\n  let found = -1;\n  for (let start = 0; start + run.length <= tokens.length; start++) found = start;\n' +
    '  return found;\n}\n' });
  const [problem] = compiled.diagnostics;
  assert.deepEqual({ file: problem.file, line: problem.line, column: problem.column, code: problem.code },
    { file: 'root/find.ts', line: 4, column: 3, code: 'forbidden-loop' });
  assert.match(problem.message, /move the arithmetic to the bound: write `start <= tokens\.length - run\.length`/);
});

test('a colon in a plain frontmatter value is a frontmatter-yaml error that says to quote the value', () => {
  const compiled = check({ 'judge.nl': '---\ndescription: What a sentence is doing: reporting a route\nargs:\n  words: string[]\n' +
    'returns: boolean\n---\nDecide.\n' });
  const [problem] = compiled.diagnostics;
  assert.deepEqual({ file: problem.file, line: problem.line, code: problem.code }, { file: 'judge.nl', line: 2, code: 'frontmatter-yaml' });
  assert.match(problem.message, /the description value contains ": ".*put the whole value in double quotes \(description: "What a sentence/);
});

const stage = (code, children) => ({ version: 'natlang.program/2', id: 'diag', kind: 'lambda_source', source: 'fixture', split: 'train',
  source_ids: ['diag'], source_groups: ['diag'], license: 'test', family: 'fixture',
  semantics: { root: 'root.nl', inputs: { note: 'a receipt for the tax return' }, expected: 'keep', oracle: 'exact', files: {
    'root.nl': '---\nargs:\n  note: string\nreturns: string\n---\nCall keep(note) and return its answer.\n',
    'root/keep.nl': '---\nargs:\n  note: string\nreturns: \'"keep" | "toss"\'\n---\nDecide whether note is worth keeping.\n' } },
  curriculum: { reference: { root: [['eval', { code, finish: true }]], ...(children ? { children } : {}) } } });

test('a reference whose script ends early quotes the last tool result: the eval error that stopped it', async () => {
  const { run } = await replayReference(stage('const keep = 1; return "keep";'), TOOLS_PROMPT);
  const detail = JSON.stringify(run.outcome.detail ?? run.outcome);
  assert.match(detail, /ended without finishing the call; its last action returned: .*keep/, detail.slice(0, 800));
});

test('an unmatched child call is named with its arguments', async () => {
  const { run } = await replayReference(stage('return await keep(note);', [{ match: ['worth keeping', 'parking ticket'], value: 'keep' }]), TOOLS_PROMPT);
  const detail = JSON.stringify(run.outcome.detail ?? run.outcome);
  assert.match(detail, /no answer for the child call keep \(its arguments: .*a receipt for the tax return/, detail.slice(0, 800));
});

// What skills/natlang-authoring says about callable-folder TypeScript and decisions (language.md, refinements.md).
test('ambient Untrusted, mixed literal unions, bare union returns, level walks and hop loops check clean', () => {
  const compiled = check({
    'types.ts': '/** A shift. */\nexport type Amount = -2 | -1 | 1 | 2 | \'reset\';\nexport type Road = { roadId: string; from: string; to: string };\n',
    'root.nl': '---\nargs:\n  roads: Road[]\n  said: Untrusted<string>\nreturns: number\n---\nCall walk(roads, said).\n',
    'root/walk.ts': 'import shift from \'./walk/shift.nl\';\nimport reading from \'./walk/reading.nl\';\nimport type { Road } from \'../types.js\';\n' +
      'export default async function walk(roads: Road[], said: Untrusted<string>): Promise<number> {\n' +
      '  const seen = new Set([roads[0]!.from]);\n  let frontier = [roads[0]!.from];\n' +
      '  for (let depth = 0; depth < roads.length && frontier.length > 0; depth++) {\n    const next: string[] = [];\n' +
      '    for (const place of frontier) for (const road of roads) if (road.from === place && !seen.has(road.to)) { seen.add(road.to); next.push(road.to); }\n' +
      '    frontier = next;\n  }\n  const parent = new Map([...seen].map(p => [p, p]));\n' +
      '  const find = (p: string): string => { let r = p; for (let hop = 0; hop < seen.size && parent.get(r) !== r; hop++) r = parent.get(r)!; return r; };\n' +
      '  for (const road of roads) parent.set(find(road.to), find(road.from));\n  let groups = 0;\n  for (const [place, up] of parent) if (place === up) groups++;\n' +
      '  const amount = await shift(said.slice(2));\n  const kind = await reading(said);\n' +
      '  return groups + (typeof amount === \'number\' ? amount : 0) + (kind === \'year\' ? 1 : 0);\n}\n',
    'root/walk/shift.nl': '---\ndescription: "How far a message moves the key: up, down or back."\nargs:\n  message: Untrusted<string>\nreturns: Amount\nreadout: decision\n---\nDecide how far message moves the key.\n',
    'root/walk/reading.nl': '---\nargs:\n  message: Untrusted<string>\nreturns: "year" | "cardinal"\nreadout: decision\n---\nDecide how the number in message is read.\n' });
  assert.deepEqual(compiled.diagnostics.filter(item => item.severity === 'error'), []);
});

test('a file of named exports binds as an object; a sliced untrusted string enters an Untrusted slot as data', async () => {
  const files = {
    'root.nl': '---\nargs:\n  note: Untrusted<string>\nreturns: string\n---\nCall pick(note).\n',
    'root/fold.ts': 'export function fold(s: string): string { return s.toLowerCase(); }\n',
    'root/pick.ts': 'import shift from \'./pick/shift.nl\';\nexport default async function pick(note: Untrusted<string>): Promise<string> {\n  return JSON.stringify(await shift(note.slice(2)));\n}\n',
    'root/pick/shift.nl': '---\nargs:\n  message: Untrusted<string>\nreturns: -1 | 1 | 2 | \'reset\'\nreadout: decision\n---\nDecide how far message moves the key.\n' };
  const record = code => ({ ...stage(code), semantics: { ...stage(code).semantics, files, inputs: { note: 'up a tone please' } } });
  const named = await replayReference(record('return [typeof fold, typeof fold.fold].join(",");'), TOOLS_PROMPT);
  assert.equal(named.run.outcome.value, 'object,function');
  const withChild = record('return await pick(note);');
  withChild.curriculum.reference.children = [{ match: ['moves the key'], value: 2 }];
  const { run, trajectory } = await replayReference(withChild, TOOLS_PROMPT);
  assert.equal(run.outcome.value, '2');
  const opening = trajectory.find(turn => /inside this call: shift/.test(String(turn.context?.[1]?.content))).context.slice(1, 4).map(m => String(m.content)).join('\n');
  assert.match(opening, /untrusted data from argument message of shift\n a tone please/);
});
