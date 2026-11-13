import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang, NatlangCallError } from '../dist/index.js';
import { readNatlangFrontmatter, parseNatlang, PATH_ONLY, NatlangSourceError } from '../dist/runtime/loader.js';
import * as natlang from '../dist/index.js';

const fn = (frontmatter, body = 'Decide.') => defineNatlang(`---\n${frontmatter}\n---\n${body}\n`);
const failure = async (runtime, call) => {
  try { await runtime.run(call); } catch (error) { return error; }
  assert.fail('the call did not fail');
};

test('a model that only ever replies in prose ends its call as no-action instead of looping for ever', async () => {
  let turns = 0;
  const runtime = createNatlangRuntime({ model: () => { turns++; return { text: 'The ticket seems urgent.', completion_tokens: 5 }; } });
  const error = await failure(runtime, () => fn('returns: \'"urgent" | "normal"\'')());
  assert.ok(error instanceof NatlangCallError, String(error));
  assert.equal(turns, 4);
  assert.equal(error.reason, 'no-action');
  assert.match(error.detail, /^model_incomplete: no_action \(4 replies in a row called no tool and returned no result; the last one said: "The ticket seems urgent\."\)/);
  assert.ok(error.callId);
});

test('a budget stop, a blocked status and a repeated wrong type are typed reasons', async () => {
  const budget = createNatlangRuntime({ model: { driver: () => ({ calls: [['eval', { code: 'return 1' }]], completion_tokens: 5 }), maxTurns: 2 } });
  assert.equal((await failure(budget, () => fn('returns: string')())).reason, 'budget-exhausted');

  const blocked = createNatlangRuntime({ model: () => ({ calls: [['return_result', { status: 'blocked', reason: 'no ticket text was given' }]] }) });
  const stop = await failure(blocked, () => fn('returns: string')());
  assert.equal(stop.reason, 'blocked');
  assert.equal(stop.detail, 'blocked: no ticket text was given');

  const failed = createNatlangRuntime({ model: () => ({ calls: [['return_result', { status: 'failed', reason: 'the instructions cannot be carried out' }]] }) });
  assert.equal((await failure(failed, () => fn('returns: string')())).reason, 'failed');

  const wrong = createNatlangRuntime({ model: () => ({ calls: [['return_result', { status: 'success', value: 'maybe' }]] }) });
  const mismatch = await failure(wrong, () => fn('returns: \'"urgent" | "normal"\'')());
  assert.equal(mismatch.reason, 'type-mismatch');
  assert.match(mismatch.detail, /^model_incomplete: repeated_failed_action/);
});

test('a failure that is not the model\'s outcome is the call\'s failure, with the original error as cause', async () => {
  const refusal = Object.assign(new Error('access denied: caller lacks model.use'), { code: 'ACCESS_DENIED' });
  const runtime = createNatlangRuntime({ model: () => { throw refusal; } });
  const error = await failure(runtime, () => fn('returns: string')());
  assert.ok(error instanceof NatlangCallError);
  assert.equal(error.reason, 'error');
  assert.equal(error.cause, refusal);
  assert.equal(error.outcome, 'failed');
  assert.equal(error.detail, 'access denied: caller lacks model.use');
  assert.ok(error.callId);

  const controller = new AbortController();
  const pending = createNatlangRuntime({ model: (_request, signal) => new Promise((_, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    queueMicrotask(() => controller.abort(new Error('stop requested by user')));
  }) });
  const cancelled = await failure({ run: call => pending.run(call, { signal: controller.signal }) }, () => fn('returns: string')());
  assert.equal(cancelled.reason, 'cancelled');
});

test('fn.on({ model }) runs one function on another configured model, leaving fn on its own', async () => {
  const seen = [];
  const answer = name => () => { seen.push(name); return { calls: [['return_result', { status: 'success', value: name }]] }; };
  const runtime = createNatlangRuntime({ model: answer('small'), models: { standard: answer('standard') } });
  const judge = fn('returns: string');
  assert.equal(await runtime.run(() => judge.on({ model: 'standard' })()), 'standard');
  assert.equal(await runtime.run(() => judge()), 'small');
  assert.deepEqual(seen, ['standard', 'small']);
  assert.throws(() => judge.on({ model: '' }), /on takes \{ model: NAME \}/);
  assert.throws(() => judge.on({ model: 'standard', turns: 3 }), /on takes \{ model: NAME \}/);
});

test('budget frontmatter bounds the calls of its function, and model names may be provider refs', async () => {
  let turns = 0;
  const runtime = createNatlangRuntime({ model: { driver: () => { turns++; return { calls: [['eval', { code: 'return 1' }]] }; }, maxTurns: 6 } });
  const error = await failure(runtime, () => fn('returns: string\nbudget: { maxTurns: 2 }')());
  assert.equal(error.reason, 'budget-exhausted');
  assert.equal(turns, 2);
  assert.deepEqual(readNatlangFrontmatter('returns: string\nbudget: { maxTurns: 8, turnTokens: 1024 }').budget, { maxTurns: 8, turnTokens: 1024 });
  assert.throws(() => fn('returns: string\nbudget: { maxTurns: 0 }'), /budget maps maxTurns, turnTokens, maxTokens, contextTokens to positive whole numbers/);
  assert.equal(parseNatlang('x.nl', '---\nreturns: string\nmodel: openai:gpt-4.1-mini\n---\nX.\n', {}, PATH_ONLY).model, 'openai:gpt-4.1-mini');
});

test('a frontmatter error names its file and the line of the offending entry', () => {
  try {
    parseNatlang('judge.nl', '---\ndescription: Judge a note.\nargs:\n  note: Nope\nreturns: string\n---\nJudge note.\n', {}, PATH_ONLY);
    assert.fail('parsed');
  } catch (error) {
    assert.ok(error instanceof NatlangSourceError);
    assert.equal(error.line, 4);
    assert.equal(error.reason, 'unknown type name Nope');
    assert.equal(error.message, 'judge.nl:4: unknown type name Nope');
  }
  const compiled = natlang.compileVirtualProject({ root: '/project', files: {
    'judge.nl': '---\nargs: { other: string, note: Nope }\nreturns: string\n---\nJudge note.\n' } }, natlang, {});
  assert.equal(compiled.ok, false);
  assert.deepEqual(compiled.diagnostics.map(({ file, line, code, message }) => ({ file, line, code, message })),
    [{ file: 'judge.nl', line: 2, code: 'callable-scope', message: 'unknown type name Nope' }]);
});
