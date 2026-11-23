// `writes:` of a directory reducer (runtime/writes.ts): the files its calls write, declared as one record type whose
// fields are paths or globs and whose field types type each file's text. Through the collector's executeProgram (RL/EI
// collection, held eval, qualification replay): the opening shows the declaration with the aliases it names (doc
// comments included), a return that wrote an undeclared file or a file whose text misses its literal type is refused
// with one sentence, and a run that keeps to it is accepted. The loader refuses malformed declarations.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { executeProgram } from '../dist/teacher/collector.js';
import { loadNamedFunction } from '../dist/runtime/loader.js';
import { virtualSourceFiles } from '../dist/runtime/virtual-project.js';

const options = runId => ({ rootSeed: 0, runId, maxTurns: 6, contextTokens: 16384, systemPrompt: 'Use the tools.' });
const ROOT = '---\nkind: directory-reducer\nargs: {}\nreturns: TallyResult\nwrites: TallyFiles\n---\n' +
  'Move every note in inbox/ to done/ and write tally.md: the number of notes moved.\n';
const TYPES = '/** How many notes were moved. */\nexport type TallyResult = number;\n' +
  '/** Every line of a written file ends with a newline. */\nexport type TallyFiles = {\n' +
  '  /** The count of moved notes, then a newline; exactly `none` and a newline when nothing moved. */\n  "tally.md": string;\n' +
  '  "done/*.md": string;\n  "status.md": "open\\n" | "closed\\n";\n};\n';
const program = (id, extra = {}) => ({ version: 'natlang.program/2', id, kind: 'lambda_source', split: 'test',
  source_ids: [id], source_groups: ['synthetic-only'],
  semantics: { root: 'tally.nl', files: { 'tally.nl': ROOT, 'types.ts': TYPES }, inputs: {}, expected: 2, oracle: 'exact',
    folder_files: { 'inbox/a.md': 'A\n', 'inbox/b.md': 'B\n' },
    expected_files: { 'done/a.md': 'A\n', 'done/b.md': 'B\n', 'tally.md': '2\n' }, ...extra } });
const MOVE = 'for (const f of await folder.files("inbox/*.md")) await f.moveTo(folder.dir("done"));\n';

async function run(id, code) {
  const openings = [], replies = [];
  let turn = 0;
  const driver = async request => {
    if (turn === 0) openings.push(String(request.messages?.[1]?.content ?? ''));
    else replies.push(String(request.messages.at(-1)?.content ?? ''));
    turn += 1;
    if (turn === 1) return { calls: [['eval', { code, finish: true }]] };
    // The repair: what an eval wrote stays written, so the stray files go and the declared ones are (re)written.
    if (turn === 2) return { calls: [['eval', { code: 'for (const p of ["notes.md", "status.md"]) if (await folder.file(p).exists()) await folder.file(p).remove();\n' +
      MOVE + 'await folder.file("tally.md").writeText("2\\n");\nreturn 2;', finish: true }]] };
    return { calls: [['return_result', { status: 'failed', reason: 'gave up after the second attempt' }]] };
  };
  const result = await executeProgram(program(id), driver, options(id));
  return { result, openings, replies };
}

test('the opening shows the declared files and the aliases they name, with their doc comments', async () => {
  const { result, openings } = await run('writes-ok', MOVE + 'await folder.file("tally.md").writeText("2\\n");\nreturn 2;');
  assert.equal(result.outcome.accepted, true, JSON.stringify(result.outcome));
  const opening = openings[0];
  assert.match(opening, /^You are inside this call: tally\(\): TallyResult\nThe files it writes: TallyFiles\n/);
  assert.match(opening, /\/\*\* How many notes were moved\. \*\/\ntype TallyResult = number;/);
  assert.match(opening, /\/\*\* Every line of a written file ends with a newline\. \*\/\ntype TallyFiles = \{/);
  assert.match(opening, /\/\*\* The count of moved notes, then a newline;[^\n]*\*\/\n\s+"tally\.md": string;/);
});

test('a file the declaration does not list is refused with one sentence, and the call can repair it', async () => {
  const { result, replies } = await run('writes-extra', MOVE + 'await folder.file("tally.md").writeText("2\\n");\n' +
    'await folder.file("notes.md").writeText("scratch\\n");\nreturn 2;');
  assert.match(replies[0], /writes-unsatisfied: notes\.md is not among the files tally writes \(tally\.md, done\/\*\.md, status\.md\)/);
  assert.equal(result.outcome.accepted, true, JSON.stringify(result.outcome));
});

test('a file whose text is not its literal type is refused', async () => {
  const { replies } = await run('writes-literal', MOVE + 'await folder.file("tally.md").writeText("2\\n");\n' +
    'await folder.file("status.md").writeText("done\\n");\nreturn 2;');
  assert.match(replies[0], /status\.md holds "done\\n", which is not a "open\\n" \| "closed\\n"/);
});

const load = (root, files = {}) => () => loadNamedFunction('/project/r.nl', virtualSourceFiles({ 'r.nl': root, ...files }));
test('the loader refuses writes on a plain function, a non-record type, a non-text field and a path outside the folder', () => {
  assert.throws(load('---\nargs: {}\nreturns: number\nwrites: { "a.md": string }\n---\nx\n'), /writes declares the files a directory reducer writes/);
  assert.throws(load('---\nkind: directory-reducer\nargs: {}\nreturns: number\nwrites: string\n---\nx\n'), /writes is one record type/);
  assert.throws(load('---\nkind: directory-reducer\nargs: {}\nreturns: number\nwrites: { "a.md": number }\n---\nx\n'), /a file holds text/);
  assert.throws(load('---\nkind: directory-reducer\nargs: {}\nreturns: number\nwrites: { "../a.md": string }\n---\nx\n'), /is not a path in the folder/);
  const record = load('---\nkind: directory-reducer\nargs: {}\nreturns: number\nwrites: Out\ntypes:\n  Out: \'{ "a.md": string, "b/*.md": "x" }\'\n---\nx\n')();
  assert.equal(record.writes, 'Out');
});
