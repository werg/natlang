// Through the collector's executeProgram (RL/EI collection, held eval, qualification replay and smoke), the two
// signature constructs held-v4's B1 authors could not use: a `.nl` helper's `ensures:` postconditions (a crisp check,
// and a judged predicate decided on the episode's own driver) and a `.nl` helper that declares a function type written
// with a promised result (`(t: string) => Promise<boolean>`, read as `(t: string) => boolean`), as a parameter and as
// its result.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { executeProgram } from '../dist/teacher/collector.js';

const callName = request => /You are inside this call: ([\w@$.:-]+)\(/.exec(String(request.messages?.[1]?.content ?? ''))?.[1];
const options = runId => ({ rootSeed: 0, runId, maxTurns: 8, contextTokens: 16384, systemPrompt: 'Use the tools.' });
const program = (id, files, inputs, expected) => ({ version: 'natlang.program/2', id, kind: 'lambda_source', split: 'test',
  source_ids: [id], source_groups: ['synthetic-only'], semantics: { root: 'r.nl', files, inputs, expected, operation: 'exact' } });
const holdsReply = (request, verdict) => Array.isArray(request.tools) && request.tools.length === 0 ?
  { text: JSON.stringify(verdict), calls: [] } : { calls: [['return_result', { status: 'success', value: verdict }]] };
const succeed = value => ({ calls: [['return_result', { status: 'success', value }]] });

const ensured = program('ensures-helper', {
  'r.nl': '---\nargs:\n  words: string[]\nreturns: string[]\n---\nShorten the list with shorten.\n',
  'r/shorten.nl': '---\nargs:\n  words: string[]\nreturns: string[]\nensures:\n  - check: "result.length <= words.length && result.every(w => words.includes(w))"\n' +
    '    message: the result keeps only words of the list\n  - predicate: a list without two words that mean the same thing\n    against: [words]\n---\n' +
    'Drop every word of words that repeats the meaning of an earlier one.\n' },
  { words: ['big', 'large', 'red'] }, ['big', 'red']);

for (const [helper, verdict, accepted, code] of [
  [['big', 'red'], true, true, null], [['big', 'red'], false, false, 'refinement-unsatisfied'], [['big', 'blue'], true, false, 'refinement-unsatisfied']])
  test(`ensures: on a helper is checked in collection (${JSON.stringify(helper)}, judge ${verdict})`, async () => {
    let judged = 0;
    const driver = async request => {
      const name = callName(request);
      if (name === 'r') return { calls: [['eval', { code: 'return await shorten(words);', finish: true }]] };
      if (name === 'holds') { judged += 1; return holdsReply(request, verdict); }
      return succeed(helper);
    };
    const run = await executeProgram(ensured, driver, options(`ensures-${helper.join('-')}-${verdict}`));
    const text = JSON.stringify(run.outcome);
    assert.doesNotMatch(text, /refinement-undecided/);
    assert.equal(run.outcome.accepted, accepted);
    if (code) assert.match(text, new RegExp(code));
    // The crisp check runs first: a result it refuses is never judged.
    if (helper.includes('blue')) assert.equal(judged, 0); else assert.ok(judged >= 1, 'the predicate was judged');
  });

const takesJudge = program('promise-param', {
  'r.nl': '---\nargs:\n  notes: string[]\nreturns: string[]\n---\nKeep the urgent notes with keep_if.\n',
  'r/keep_if.nl': '---\nargs:\n  notes: string[]\n  judge: "(note: string) => Promise<boolean>"\nreturns: string[]\n---\nKeep the notes judge accepts.\n' },
  { notes: ['server down', 'lunch menu'] }, ['server down']);

test('a .nl helper whose parameter is a Promise-returning function type takes an inline nl judge', async () => {
  const seen = [];
  const driver = async request => {
    const name = callName(request); seen.push(name);
    if (name === 'r') return { calls: [['eval', { code: 'return await keep_if(notes, nl<boolean>`Decide whether note is urgent.`);', finish: true }]] };
    if (name === 'keep_if') {
      // One turn: the eval finishes the call. A second turn means it failed; show why.
      if (seen.filter(item => item === 'keep_if').length > 1) return { calls: [['return_result', { status: 'failed', reason: String(request.messages.at(-1)?.content) }]] };
      return { calls: [['eval', { code: 'const kept = []; for (const note of notes) if (await judge(note)) kept.push(note); return kept;', finish: true }]] };
    }
    const urgent = /server down/.test(JSON.stringify(request.messages.slice(1)));
    return succeed(urgent);
  };
  const run = await executeProgram(takesJudge, driver, options('promise-param'));
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome).slice(0, 1500) + ' ' + seen.join(','));
});

const makesJudge = program('promise-result', {
  'r.nl': '---\nargs:\n  notes: string[]\n  rubric: string\nreturns: string[]\n---\nKeep the notes that meet rubric, with the judge judge_for makes.\n',
  'r/judge_for.nl': '---\nargs:\n  rubric: string\nreturns: "(note: string) => Promise<boolean>"\n---\nMake a judge of whether a note meets rubric.\n' },
  { notes: ['server down', 'lunch menu'], rubric: 'urgent' }, ['server down']);

test('a .nl helper whose result is a Promise-returning function type returns a callable judge', async () => {
  const seen = [];
  const driver = async request => {
    const name = callName(request); seen.push(name);
    if (name === 'r') return { calls: [['eval', { code: 'const judge = await judge_for(rubric);\nconst kept = [];\n' +
      'for (const note of notes) if (await judge(note)) kept.push(note);\nreturn kept;', finish: true }]] };
    if (name === 'judge_for') return { calls: [['eval', { code: 'return nl.with<boolean>({ rubric })`Decide whether note meets rubric.`;', finish: true }]] };
    return succeed(/server down/.test(JSON.stringify(request.messages.slice(1))));
  };
  const run = await executeProgram(makesJudge, driver, options('promise-result'));
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome).slice(0, 1500) + ' ' + seen.join(','));
});
