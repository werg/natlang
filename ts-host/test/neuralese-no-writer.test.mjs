import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { executeProgram } from '../dist/teacher/collector.js';
import { NO_NEURALESE_WRITER } from '../dist/native/neuralese.js';

const call = (name, args) => ({ calls: [[name, args]] });

// teacher-gen-20261119/gap-v1: a remote teacher without a Neuralese writer got `neuralese-unsupported-backend` as the
// tool result of every Neuralese<T> return and retried it until the 40-call episode budget. No action can repair a
// runtime without a writer, so the run ends at the first such boundary.
test('a Neuralese result on a runtime without a writer ends the run instead of becoming a tool result', async () => {
  let requests = 0;
  const runtime = createNatlangRuntime({ calls: false, model: async () => { requests++; return call('return_result', { status: 'success', value: 'memo' }); } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {}\nreturns: Neuralese<string>\n---\nReturn a memo.\n' }, 'run.nl');
  await assert.rejects(runtime.run(() => fn()), error => String(error?.message ?? error).includes(NO_NEURALESE_WRITER));
  assert.equal(requests, 1);
});

test('the same fault inside eval finish and in a nested call ends the parent run too', async () => {
  for (const finishInEval of [true, false]) {
    let requests = 0;
    const record = { version: 'natlang.program/2', id: 'no-writer-nested', kind: 'lambda_source', split: 'train',
      semantics: { root: 'solve.nl', files: {
        'solve.nl': '---\nargs: {}\nreturns: string\n---\nCall memo, then finish.\n',
        'solve/memo.nl': '---\nargs: {}\nreturns: Neuralese<{ facts: string[] }>\n---\nWrite the facts.\n' }, inputs: {}, expected: 'done' } };
    const driver = async () => {
      requests++;
      if (requests === 1) return call('eval', { code: 'const m = await memo(); return "done";', finish: true });
      return finishInEval ? call('eval', { code: 'return { facts: ["a"] };', finish: true }) :
        call('return_result', { status: 'success', value: { facts: ['a'] } });
    };
    let failure;
    try {
      const result = await executeProgram(record, driver, { systemPrompt: 'Use the tools.', contextTokens: 8192, maxTurns: 8,
        rootSeed: 3, runId: `no-writer-${finishInEval}` });
      failure = JSON.stringify(result.outcome);
    } catch (error) { failure = String(error?.message ?? error); }
    assert.ok(failure.includes(NO_NEURALESE_WRITER), failure);
    assert.equal(requests, 2, 'neither the child nor the parent is asked again');
  }
});

test('read_code shows a type alias declared in the callable folder types.ts', async () => {
  const results = [];
  let step = 0;
  const calls = [['read_code', { name: 'Money' }], ['return_result', { status: 'success', value: 'string' }]];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    if (step) results.push(String(request.messages.at(-1).content));
    return { calls: [calls[step++] ?? ['return_result', { status: 'failed', reason: 'no more calls' }]] };
  } });
  const fn = loadVirtualNatlang({
    'explore.nl': '---\nargs: {}\nreturns: string\n---\nWhat is the declared type of minorUnits of Money?\n',
    'explore/types.ts': 'export type Money = { minorUnits: string; currency: string; scale: number };\n',
    'explore/session.nl': '---\nargs: { amount: string }\nreturns: string\n---\nRecord the amount.\n',
  }, 'explore.nl');
  assert.equal(await runtime.run(() => fn()), 'string');
  assert.match(results[0], /type Money = \{ minorUnits: string; currency: string; scale: number \};/);
  assert.doesNotMatch(results[0], /no-such-function/);
});

test('read_code shows a type alias of a program bound with uses (the natlang-codebase field-type task layout)', async () => {
  const results = [];
  let step = 0;
  const calls = [['read_code', { name: 'Money' }], ['return_result', { status: 'success', value: 'string' }]];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    if (step) results.push(String(request.messages.at(-1).content));
    return { calls: [calls[step++] ?? ['return_result', { status: 'failed', reason: 'no more calls' }]] };
  } });
  const fn = loadVirtualNatlang({
    'natlang.json': '{}',
    'task/explore_program.nl': '---\nuses: [explore_program/session]\nargs: {}\nreturns: "string"\n---\n' +
      'In the program in your callable folder, what is the declared type of the field minorUnits of Money?\n',
    'explore_program/types.ts': 'export type ID = string; export type Money = { minorUnits: string; currency: string; scale: number };\n',
    'explore_program/session.nl': '---\nargs: { amount: string }\nreturns: string\n---\nRecord the amount.\n',
  }, 'task/explore_program.nl');
  assert.equal(await runtime.run(() => fn()), 'string');
  assert.match(results[0], /type Money = \{ minorUnits: string; currency: string; scale: number \};/);
});
