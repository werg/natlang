import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { OPERATOR_FRAMES, viewSiteMessages } from '../dist/native/operator-frames.js';
import { VIEW_PROMPT } from '../dist/builtin/index.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const ID = 'nzs1_' + 'a'.repeat(52);

async function run(compaction) {
  const captures = [], calls = [], operatorCalls = [];
  const port = { dialect: 'native-state/test', forcedText: true, async write() { throw new Error('unused'); },
    async capture(tokens, options) {
      captures.push({ tokens, options });
      return { id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 3, width: 8,
        producer: { native_source_capture: { segments: [], payload_start: 5, payload_stop: 8 } } };
    } };
  const driver = Object.assign(async request => {
    if (request.invocation_id?.includes('#operator-')) {
      operatorCalls.push(request);
      return { text: 'value: 20000 lines of "line of data"', completion_tokens: 4, raw_response: {
        x_natlang_rl_capture: { prompt: [{ kind: 'tokens', token_ids: [1, 2, 3, 4] }], generated_inputs: [
          { token_id: 9, input_role: 'ordinary' }, { token_id: 10, input_role: 'ordinary' }] } } };
    }
    calls.push(request);
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: driver,
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', port, ...(compaction ? { compaction } : {}) } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n' }, 'run.nl');
  let error;
  try { assert.equal(await runtime.run(() => fn('line of data\n'.repeat(20000))), 7); } catch (caught) { error = caught; }
  return { captures, operatorCalls, error, request: calls.length ? JSON.stringify(calls[0].messages) : '' };
}

test('compact-params is an operator call at view\'s site plus the in-context capture of its reply', async () => {
  const { captures, operatorCalls, error } = await run('nl');
  assert.equal(operatorCalls.length, 1);
  const [call] = operatorCalls;
  assert.match(call.invocation_id, /#operator-1$/);
  assert.equal(call.max_tokens, 768);
  assert.deepEqual(call.tools, []);
  assert.equal(call.messages[0].content, VIEW_PROMPT);
  assert.match(call.messages[1].content, /^instructions:\nThe call that receives this argument as value \(string\) answers: Count the lines of value\.\n\nvalue:\n"line of data/);
  assert.deepEqual(captures, [{ tokens: { prompt: [1, 2, 3, 4], reply: [9, 10] }, options: { type: 'Neuralese<string>' } }]);
  // The fake state has no archived artifact, so the model turn refuses it for provenance: it was placed in the request.
  assert.match(String(error?.message), new RegExp(`${ID} is unavailable for provenance`));
});

test('the crisp default keeps the text cut and never calls the operator', async () => {
  const { operatorCalls, captures, request, error } = await run(undefined);
  assert.equal(error, undefined);
  assert.equal(operatorCalls.length + captures.length, 0);
  assert.ok(!request.includes(ID));
  assert.match(request, /cut off/);
});

test('view\'s site and the consumer frames equal the Python definitions', async () => {
  assert.equal(OPERATOR_FRAMES['compact-history'].close, '\n\nContinue from the most recent steps.');
  assert.deepEqual(viewSiteMessages('V', 'I'), [{ role: 'system', content: VIEW_PROMPT },
    { role: 'user', content: 'instructions:\nI\n\nvalue:\nV' }]);
  assert.deepEqual(viewSiteMessages('V')[1], { role: 'user', content: 'value:\nV' });
});
