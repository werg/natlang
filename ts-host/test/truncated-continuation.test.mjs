import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { executeProgram, completedActionProgress } from '../dist/teacher/collector.js';
import { interpreter, lambda } from './support/natlang.mjs';

const truncated = index => ({ calls: [], text: `Synthetic incomplete reply ${index}`, truncated: true,
  completion_tokens: 8, raw_response: { choices: [{ finish_reason: 'length', message: { content: 'partial' } }] } });
const done = { calls: [['return_result', { status: 'success', value: 1 }]], completion_tokens: 1 };
async function runReplies(replies, options) {
  const requests = [];
  const agent = new NativeToolAgent(request => { requests.push(structuredClone(request)); return replies(requests.length); }, options);
  const result = await interpreter({ agent: session => agent.run(session) }).run(lambda({ type: '() => number', instructions: 'Return one.' }));
  return { result, requests };
}

test('completed truncated no-action replies consume the declared repair budget with unchanged recovery text', async () => {
  const { result, requests } = await runReplies(index => truncated(index), { maxFailureRepairs: 1 });
  assert.equal(requests.length, 2);
  assert.equal(result.outcome.kind, 'quiesced');
  assert.match(result.outcome.detail, /^model_incomplete: truncated_without_action/);
  assert.match(requests[1].messages.at(-1).content, /Your reply was cut off .* before any tool call\. Take the next step with one tool call\./);
});

test('executed successful eval resets the no-action failures, malformed proposals cannot', async () => {
  const replies = [truncated(0), { calls: [['eval', { code: 'const computed = 1; computed;' }]], completion_tokens: 1 }, truncated(2), done];
  const good = await runReplies(index => replies[index - 1], { maxFailureRepairs: 1 });
  assert.equal(good.requests.length, 4); assert.equal(good.result.value, 1);
  const bad = await runReplies(index => index === 1 ? truncated(0) : { calls: [['unknown_tool', {}]], completion_tokens: 1 }, { maxFailureRepairs: 1 });
  assert.equal(bad.requests.length, 2); assert.equal(bad.result.outcome.kind, 'quiesced');
  assert.match(bad.result.outcome.detail, /repair limit reached/);
});

test('no new default repair cap; successful ordinary results and a pending decode remain allowed', async () => {
  const unlimited = await runReplies(index => index < 5 ? truncated(index) : done, { maxTurns: 8 });
  assert.equal(unlimited.result.value, 1); assert.equal(unlimited.requests.length, 5);
  const text = await runReplies(index => index === 1 ? truncated(0) : { text: '1', completion_tokens: 1 }, { maxFailureRepairs: 1 });
  assert.equal(text.result.value, 1);
  let release, calls = 0;
  const pending = runReplies(() => { calls++; return new Promise(resolve => { release = () => resolve(done); }); }, { maxFailureRepairs: 0 });
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(calls, 1); release();
  assert.equal((await pending).result.value, 1);
});

test('collector progress receipts come from actual successful execution, excluding rejected proposals', async () => {
  const record = { version: 'natlang.program/2', id: 'synthetic-progress', kind: 'lambda_source', split: 'test',
    source_ids: ['synthetic-progress'], source_groups: ['synthetic-only'], semantics: { root: 'one.nl',
      files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' }, inputs: {}, expected: 1, operation: 'exact' } };
  const receipts = []; let turns = 0;
  const run = await executeProgram(record, async request => {
    receipts.push(completedActionProgress(request.invocation_id)); turns++;
    if (turns === 1) return { calls: [['unknown_tool', {}]], completion_tokens: 1 };
    if (turns === 2) return { calls: [['eval', { code: 'return 1;' }]], completion_tokens: 1 };
    return done;
  }, { rootSeed: 0, runId: 'synthetic-progress-run', maxTurns: 5, contextTokens: 16384 });
  assert.equal(run.outcome.accepted, true); assert.equal(receipts.length, 3);
  assert.equal(receipts[1].successful_actions, receipts[0].successful_actions);
  assert.equal(receipts[2].successful_actions, receipts[1].successful_actions + 1);
  assert.equal(receipts[2].invocation_id, 'synthetic-progress-run');
  assert.equal(completedActionProgress('not-running'), undefined);
});
