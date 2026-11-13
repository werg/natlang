import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, nodePlatform } from '../dist/index.js';

/** Run root.nl with scripted root tool calls; returns each tool result and the call's value. */
async function script(calls, returns = 'string', platform) {
  const results = [];
  let step = 0;
  const model = async request => {
    if (step) results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' }, ...(platform ? { platform } : {}) });
  const fn = loadVirtualNatlang({ 'root.nl': `---\nargs: {}\nreturns: ${JSON.stringify(returns)}\n---\nSay hello.\n` }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn()); } catch {}
  return { results, value };
}

test('return_result accepts a complete later record-union arm and rejects a genuinely missing field', async () => {
  const returns = '{ chains: ({ cause: string, invoiceId: string, creditId: string, refundIds: string[] } | { cause: string, creditId: string, refundIds: string[] })[] }';
  const incomplete = { chains: [{ cause: 'credit', creditId: 'C2' }] };
  const complete = { chains: [{ cause: 'invoice', invoiceId: 'I1', creditId: 'C1', refundIds: [] },
    { cause: 'credit', creditId: 'C2', refundIds: ['R1'] }] };
  const { results, value } = await script([
    ['return_result', { status: 'success', value: incomplete }],
    ['return_result', { status: 'success', value: complete }],
  ], returns);
  assert.match(results[0], /return_result needs a complete/);
  assert.deepEqual(value, complete);
});

test('a value sent as blocked gets stop-shape repair without unconditional success advice', async () => {
  const { results, value } = await script([
    ['return_result', { status: 'blocked', value: 'hello' }],
    ['return_result', { status: 'success', value: 'hello' }]]);
  assert.match(results[0], /keep status "blocked", omit value/);
  assert.match(results[0], /success" only if the task is actually complete/);
  assert.equal(value, 'hello');
});

test('a blocked call without a value is asked only for its reason', async () => {
  const { results } = await script([['return_result', { status: 'blocked' }]]);
  assert.match(results[0], /saying what is missing/);
  assert.doesNotMatch(results[0], /use status "success"/);
});

test('read_code documents the built-ins of eval, which cannot be edited', async () => {
  const { results } = await script([['read_code', { name: 'nl' }], ['read_code', { name: 'iterateOn' }],
    ['edit_code', { name: 'nl', find: 'a', replace_with: 'b' }]]);
  assert.match(results[0], /^nl: create a natural-language function/);
  assert.match(results[1], /^iterateOn: repeat a step/);
  assert.match(results[2], /nl is built into eval and cannot be changed/);
});

test('read_code says when a name is one of the tools, not a function of the program', async () => {
  const { results } = await script([['read_code', { name: 'read_page' }]]);
  assert.match(results[0], /^Native tool definition \(not program function source\): read_page/);
  assert.match(results[0], /Invoke this as a separate tool action; it is not callable from eval\./);
});

test('a result sent under another argument name is told to use value', async () => {
  const { results, value } = await script([
    ['return_result', { status: 'success', verdict: 'hello' }],
    ['return_result', { status: 'success', value: 'hello' }]]);
  assert.equal(results[0].split('\n')[1], 'return_result needs the string itself as its value argument.');
  assert.equal(value, 'hello');
});

test('return_result value rejections render in each wording; the sentence wordings carry no diagnostic code', async () => {
  const { RETURN_ERROR_TEXTS, returnValueRejection, returnErrorText } = await import('../dist/native/return-errors.js');
  const { Reject } = await import('../dist/native/values.js');
  const render = (variant, problem) => new Reject([returnValueRejection(variant, problem, 'Judgment')]).message;
  const misplaced = { kind: 'misplaced', fields: ['verdict'] };
  assert.equal(render('legacy', misplaced), 'value: bad-action, expected a Judgment or a complete typed result already staged in this call');
  assert.equal(render('field', misplaced), 'value: bad-action, expected the result in the value argument (this return_result put it in "verdict")');
  assert.equal(render('sentence', misplaced), 'return_result needs the Judgment itself as its value argument.');
  assert.equal(render('steps', { kind: 'incomplete' }),
    'Call return_result again with status "success" and value set to a complete Judgment, every field filled, or return it from an eval.');
  for (const variant of ['sentence', 'steps'])
    for (const problem of [misplaced, { kind: 'missing' }, { kind: 'incomplete' }])
      assert.doesNotMatch(render(variant, problem), /bad-action|verdict|^value:/);
  assert.deepEqual([...RETURN_ERROR_TEXTS], ['legacy', 'field', 'sentence', 'steps']);
  assert.throws(() => returnErrorText('loud'), /returnErrorText is one of legacy, field, sentence, steps/);
  // The kernel reads no environment: the Node platform supplies NATLANG_RETURN_ERROR_TEXT, and a platform value can set it.
  const previous = process.env.NATLANG_RETURN_ERROR_TEXT;
  process.env.NATLANG_RETURN_ERROR_TEXT = 'steps';
  try {
    const { results, value } = await script([
      ['return_result', { status: 'success', verdict: 'hello' }],
      ['return_result', { status: 'success', value: 'hello' }]]);
    assert.match(results[0], /Call return_result again with status "success" and value set to the string/);
    assert.equal(value, 'hello');
    const legacy = await script([['return_result', { status: 'success', verdict: 'hello' }], ['return_result', { status: 'success', value: 'hello' }]],
      'string', { ...nodePlatform(), returnErrorText: 'legacy' });
    assert.match(legacy.results[0], /value: bad-action/);
  } finally {
    if (previous === undefined) delete process.env.NATLANG_RETURN_ERROR_TEXT; else process.env.NATLANG_RETURN_ERROR_TEXT = previous;
  }
});

test('a structured result sent as JSON or JS-literal text is read as the data it spells out', async () => {
  const returns = '{ facts: string[], count: number }';
  const json = await script([['return_result', { status: 'success', value: '{"facts": ["a", "b"], "count": 2}' }]], returns);
  assert.deepEqual(json.value, { facts: ['a', 'b'], count: 2 });
  const js = await script([['return_result', { status: 'success', value: "{ facts: ['a', \"b\",], count: -2, }" }]], returns);
  assert.deepEqual(js.value, { facts: ['a', 'b'], count: -2 });
  const list = await script([['return_result', { status: 'success', value: "[{ id: 'x' }]" }]], '{ id: string }[]');
  assert.deepEqual(list.value, [{ id: 'x' }]);
});

test('structured-result text that does not check names the fix in one sentence, never running it', async () => {
  const returns = '{ facts: string[] }';
  globalThis.__ranLiteral = false;
  const { results, value } = await script([
    ['return_result', { status: 'success', value: '{ facts: [x] }' }],
    ['return_result', { status: 'success', value: '{ facts: [(() => { globalThis.__ranLiteral = true; return "a"; })()] }' }],
    ['return_result', { status: 'success', value: 'The facts are a and b.' }],
    ['return_result', { status: 'success', value: '{ facts: [1] }' }],
    ['return_result', { status: 'success', value: '{ facts: ["a"], extra: 1 }' }],
    ['return_result', { status: 'success', value: '{ "__proto__": { "facts": ["p"] } }' }],
    ['return_result', { status: 'success', value: { facts: ['a'] } }],
  ], returns);
  const sentence = /return_result needs the \{ facts: string\[\] \} itself as its value argument, written as an object in the arguments rather than as text\./;
  for (const i of [0, 1, 2]) assert.match(results[i], sentence);
  assert.equal(globalThis.__ranLiteral, false);
  assert.match(results[3], /return\/facts\/0: type-mismatch, expected string, got 1/);
  assert.match(results[4], /return\/extra: unknown-field/);
  assert.doesNotMatch(results[5], /Returned/);
  assert.deepEqual(value, { facts: ['a'] });
  delete globalThis.__ranLiteral;
});

test('a string result is still text, even when it reads as a literal', async () => {
  const { value } = await script([['return_result', { status: 'success', value: '{ facts: [1] }' }]], 'string');
  assert.equal(value, '{ facts: [1] }');
});
