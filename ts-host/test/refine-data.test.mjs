import assert from 'node:assert/strict';
import { test } from 'node:test';
import { judgeMessages, labelPairs, runStages, stageFiles } from '../scripts/refine-data/lib.mjs';

/** A scripted model: `reply(stage text, request)` gives the value returned by return_result. No model is loaded. */
const scripted = reply => Object.assign(async request => {
  const text = request.messages.map(message => typeof message.content === 'string' ? message.content : JSON.stringify(message.content)).join('\n');
  return { calls: [['return_result', { status: 'success', value: reply(text) }]] };
}, {});

test('nearMiss runs as a natlang function through the runtime and returns the edited value', async () => {
  const seen = [];
  const driver = scripted(text => {
    seen.push(text);
    return { edited: 'Order shipped yesterday and the parcel is on its way to you today.', edit: 'lengthened the line beyond the limit' };
  });
  const rows = [{ id: 'a', args: { predicate: 'one line of at most 60 characters', value: 'Order shipped yesterday.' } }];
  const [result] = await runStages({ stage: 'nearMiss', rows, driver, runtimeOptions: { calls: false } });
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.value.edit, 'lengthened the line beyond the limit');
  assert.match(result.value.edited, /^Order shipped/);
  assert.ok(seen.some(text => text.includes('one line of at most 60 characters') && text.includes('Order shipped yesterday.')),
    'the predicate and the value reach the model');
});

test('verifyNearMiss and exemplify return their typed results', async () => {
  const verify = await runStages({ stage: 'verifyNearMiss', runtimeOptions: { calls: false }, driver: scripted(() => ({ original_holds: true, edited_holds: false, minimal: true, reason: 'the length' })),
    rows: [{ id: 'v', args: { predicate: 'p', original: 'a', edited: 'b' } }] });
  assert.deepEqual(verify[0].value, { original_holds: true, edited_holds: false, minimal: true, reason: 'the length' });
  const examples = await runStages({ stage: 'exemplify', runtimeOptions: { calls: false }, driver: scripted(() => ({ values: ['x', 'y'] })),
    rows: [{ id: 'e', args: { predicate: 'p', base: 'string', slot: 'Outgoing.subject', count: 2 } }] });
  assert.deepEqual(examples[0].value.values, ['x', 'y']);
});

test('a resumed stage skips finished ids', async () => {
  let calls = 0;
  const driver = scripted(() => { calls++; return { edited: 'b', edit: 'e' }; });
  const rows = [1, 2, 3].map(n => ({ id: `r${n}`, args: { predicate: 'p', value: 'a' } }));
  const results = await runStages({ stage: 'nearMiss', rows, driver, done: new Set(['r1']), runtimeOptions: { calls: false } });
  assert.deepEqual(results.map(result => result.id).sort(), ['r2', 'r3']);
  assert.equal(calls, 2);
});

test('labelPairs scores with the runtime judge prompt and reports P(true)', async () => {
  const prompts = [];
  const decide = async request => {
    prompts.push(request);
    const value = /<<<value\n([^]*?)\nvalue>>>/.exec(String(request.messages.at(-1).content))[1];
    return { log_probs: value.length <= 5 ? [Math.log(0.9), Math.log(0.1)] : [Math.log(0.2), Math.log(0.8)] };
  };
  const pairs = [{ id: 'p1', value: 'short', predicate: 'brief' }, { id: 'p2', value: 'much too long', predicate: 'brief' }];
  const results = await labelPairs({ pairs, decide, teacher: 'teacher:test' });
  const byId = Object.fromEntries(results.map(result => [result.id, result]));
  assert.ok(Math.abs(byId.p1.p_true - 0.9) < 1e-9);
  assert.ok(Math.abs(byId.p2.p_true - 0.2) < 1e-9);
  assert.deepEqual(prompts[0].options, ['true', 'false']);
  assert.equal(byId.p1.teacher, 'teacher:test');
});

test('a scorer failure is recorded on the pair so a later run retries it', async () => {
  const decide = async () => { throw new Error('decision-unsupported: no prompt_logprobs'); };
  const [result] = await labelPairs({ pairs: [{ id: 'x', value: 'v', predicate: 'p' }], decide, teacher: 't' });
  assert.match(result.error, /decision-unsupported/);
  assert.equal(result.p_true, undefined);
});

test('the stage files exist and judgeMessages renders non-strings as canonical JSON', async () => {
  assert.deepEqual(Object.keys(stageFiles()).sort(), ['exemplify.nl', 'nearMiss.nl', 'types.ts', 'verifyNearMiss.nl']);
  const messages = await judgeMessages({ b: 1, a: [2] }, '  a   record ');
  assert.match(messages[1].content, /<<<value\n\{"a":\[2\],"b":1\}\nvalue>>>/);
  assert.match(messages[1].content, /Property: the value is a record\n/);
});
