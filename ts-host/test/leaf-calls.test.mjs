import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { LEAF_SYSTEM_PROMPT } from '../dist/native/prompt.js';
import { promptPieces } from '../dist/native/system-prompts.js';

const ANSWER = '---\nargs: { input: string }\nreturns: "billing" | "technical" | "other"\n---\nClassify input.\n';
const toolNames = request => (request.tools ?? []).map(tool => tool.function.name);

/** A driver that records each request and answers with the next scripted reply. */
function scripted(...replies) {
  const requests = [];
  const driver = async request => { requests.push(request); return replies.shift() ?? { text: 'done' }; };
  return { driver, requests };
}

test('leafCalls direct: a data-only call answers from the short prompt, its arguments shown, with only return_result', async () => {
  const { driver, requests } = scripted({ text: 'billing' });
  const runtime = createNatlangRuntime({ model: { driver, leafCalls: 'direct' }, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'answer.nl': ANSWER }, 'answer.nl');
  assert.equal(await runtime.run(() => fn('I was charged twice')), 'billing');
  assert.equal(requests.length, 1);
  const [system, user] = requests[0].messages;
  assert.equal(system.content, LEAF_SYSTEM_PROMPT);
  assert.match(user.content, /^You are inside this call: answer\(input: string\)/);
  assert.match(user.content, /Arguments:\ninput: string = "I was charged twice"/);
  assert.doesNotMatch(user.content, /eval/);
  assert.deepEqual(toolNames(requests[0]), ['return_result']);
  assert.equal(requests[0].messages.length, 2, 'no synthetic eval exchange in the opening');
  assert.ok(promptPieces().some(piece => piece.id === 'leaf' && piece.text === LEAF_SYSTEM_PROMPT), 'the leaf prompt is a registered piece');
});

test('a leaf call returns structured JSON, repairs an invalid reply, and reports blocked through return_result', async () => {
  const record = '---\nargs: { note: string }\nreturns: "{ name: string; age: number }"\n---\nExtract the person in note.\n';
  const fn = loadVirtualNatlang({ 'person.nl': record }, 'person.nl');
  const repaired = scripted({ text: 'Ada is 36.' }, { text: '```json\n{"name": "Ada", "age": 36}\n```' });
  const runtime = createNatlangRuntime({ model: { driver: repaired.driver, leafCalls: 'direct' }, seed: { mode: 'backend' } });
  assert.deepEqual(await runtime.run(() => fn('Ada, 36, mathematician')), { name: 'Ada', age: 36 });
  assert.equal(repaired.requests.length, 2, 'prose is not a result: the call asks again');
  const blocked = scripted({ calls: [['return_result', { status: 'blocked', reason: 'note names nobody' }]] });
  const strict = createNatlangRuntime({ model: { driver: blocked.driver, leafCalls: 'direct' }, seed: { mode: 'backend' } });
  await assert.rejects(strict.run(() => fn('the weather is fine')), /note names nobody/);
});

test('calls that need the interpreter keep the full surface under leafCalls direct, and tools is the default', async () => {
  const withHelper = loadVirtualNatlang({ 'step.nl': '---\nargs: {}\nreturns: boolean\n---\nAsk child.',
    'step/child.nl': '---\nargs: {}\nreturns: boolean\n---\nDecide.' }, 'step.nl');
  const helper = scripted({ calls: [['return_result', { status: 'success', value: true }]] });
  await createNatlangRuntime({ model: { driver: helper.driver, leafCalls: 'direct' }, seed: { mode: 'backend' } }).run(() => withHelper());
  assert.ok(toolNames(helper.requests[0]).includes('eval'), 'a function with callable children is not a leaf');
  assert.notEqual(helper.requests[0].messages[0].content, LEAF_SYSTEM_PROMPT);

  const fileArg = loadVirtualNatlang({ 'read.nl': '---\nargs: { doc: FileHandle }\nreturns: string\n---\nSummarize doc.\n' }, 'read.nl');
  const files = scripted({ calls: [['return_result', { status: 'success', value: 'x' }]] });
  const runtime = createNatlangRuntime({ model: { driver: files.driver, leafCalls: 'direct' }, seed: { mode: 'backend' } });
  await runtime.run(() => fileArg(undefined)).catch(() => {});
  if (files.requests.length) assert.ok(toolNames(files.requests[0]).includes('eval'), 'a FileHandle parameter needs the interpreter');

  const plain = scripted({ text: 'billing' });
  const fn = loadVirtualNatlang({ 'answer.nl': ANSWER }, 'answer.nl');
  await createNatlangRuntime({ model: { driver: plain.driver }, seed: { mode: 'backend' } }).run(() => fn('I was charged twice'));
  assert.ok(toolNames(plain.requests[0]).includes('eval'), 'leafCalls defaults to tools');
});
