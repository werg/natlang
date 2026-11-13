// Decided model-facing behaviour (2026-11-12, plans/MODEL_FACING_CHANGES.md): every sampled call of a reply stays in the
// history, and every turn offers the same tools.
import assert from 'node:assert/strict';
import { test } from 'node:test';
const { createNatlangRuntime, loadVirtualNatlang } = await import('../dist/index.js');

test('calls after a failed call of the same reply stay in the history, each answered as not run', async () => {
  let seen;
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    const last = request.messages.at(-1);
    if (last.role === 'tool' && last.tool_call_id !== 'scope_0') {
      seen ??= request.messages;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { calls: [['eval', { code: 'throw new Error("first fails")' }], ['eval', { code: '1 + 1' }],
      ['list_files', {}]] };
  } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  const reply = seen.findLast(message => message.role === 'assistant');
  assert.equal(reply.tool_calls.length, 3, 'every sampled call stays in the history');
  const answers = seen.filter(message => message.role === 'tool' && reply.tool_calls.some(call => call.id === message.tool_call_id));
  assert.equal(answers.length, 3);
  assert.match(String(answers[0].content), /first fails/);
  assert.equal(String(answers[1].content), 'Not run: an earlier call in this reply failed; call it again if you still need it.');
  assert.doesNotMatch(String(answers[2].content), /^Not run/, 'a read-only call still runs');
});

test('the last turn offers every tool and, on a Neuralese driver, forces return_result by template', async () => {
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    return requests.length === 1 ? { calls: [['eval', { code: '1 + 1' }]] } : { calls: [['return_result', { status: 'success', value: 'hi' }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: { maxTurns: 2, driver } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
  try { await runtime.run(() => fn('Ada')); } catch {}
  const names = requests.map(request => request.tools.map(tool => tool.function.name).join());
  assert.ok(requests.length >= 2 && names.every(list => list === names[0]), JSON.stringify(names));
  assert.equal(requests[0].template, undefined);
  assert.deepEqual(requests.at(-1).template, { call: 'return_result', arguments: {}, argument: 'status', value: 'decode', optional: true });
  assert.match(String(requests.at(-1).messages.at(-1).content), /last turn/);
});

// History fidelity (2026-11-13, plans/MODEL_FACING_CHANGES.md): the next prompt carries each reply as it was sampled.
const greet = () => loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: string\n---\nGreet name.\n' }, 'root.nl');
const afterOpening = request => { const last = request.messages.at(-1); return last.role !== 'tool' || last.tool_call_id !== 'scope_0'; };

test('what a reply says before its tool calls stays in the history', async () => {
  let seen;
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    if (request.messages.some(message => message.role === 'assistant' && message.content === 'I will add first.\n\n')) {
      seen ??= request.messages;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { text: 'I will add first.\n\n', reasoning: 'Add, then answer.', calls: [['eval', { code: '1 + 1' }]] };
  } });
  assert.equal(await runtime.run(() => greet()('Ada')), 'done');
  const reply = seen.findLast(message => message.role === 'assistant');
  assert.equal(reply.content, 'I will add first.\n\n', 'whitespace included');
  assert.equal(reply.reasoning_content, 'Add, then answer.');
  assert.equal(reply.tool_calls.length, 1);
});

test('a call that is not a tool keeps its sampled arguments and gets its own error', async () => {
  let seen;
  const raw = [{ id: 'c0', type: 'function', function: { name: 'transcript.search', arguments: '{"text":"French"}' } }];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    if (afterOpening(request) && request.messages.at(-1).role === 'tool') {
      seen ??= request.messages;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { calls: [['transcript.search', { text: 'French' }]], raw_calls: raw };
  } });
  assert.equal(await runtime.run(() => greet()('Ada')), 'done');
  const reply = seen.findLast(message => message.role === 'assistant');
  assert.deepEqual(reply.tool_calls, raw, 'the call is in the history exactly as the server reported it');
  assert.match(String(seen.at(-1).content), /transcript\.search is not a tool/);
});

test('a reply whose tool call could not be read stays in the history with what was wrong', async () => {
  const leaked = 'Calling it.\n<tool_call>\n<function=eval>\n<parameter=code>\n1 + 1\n';
  const seen = [];
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request => {
    const last = request.messages.at(-1);
    if (last.role === 'user' && /could not be read|was not closed|left a parameter open/.test(String(last.content))) {
      seen.push(request.messages);
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    }
    return { calls: [], text: leaked, reasoning: 'Run it.',
      malformed_call: 'The tool call in your last reply left a parameter open or gave one twice, so nothing ran; send the call again, complete.' };
  } });
  assert.equal(await runtime.run(() => greet()('Ada')), 'done', 'the markup is never the text answer');
  const [reply, feedback] = seen[0].slice(-2);
  assert.deepEqual([reply.role, reply.content, reply.reasoning_content], ['assistant', leaked, 'Run it.']);
  assert.equal(feedback.content, 'The tool call in your last reply left a parameter open or gave one twice, so nothing ran; send the call again, complete.');
});

test('a record result sent as JS-literal text is one return_result call in the history, and is the result', async () => {
  const value = '{ action: "order", customer: "Wen", pence: 280 }';
  const runtime = createNatlangRuntime({ seed: { mode: 'backend' }, model: async request =>
    ({ calls: [['return_result', { status: 'success', value }]] }) });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { name: string }\nreturns: { action: string, customer: string, pence: number }\n---\nRead the order.\n' }, 'root.nl');
  assert.deepEqual(await runtime.run(() => fn('Ada')), { action: 'order', customer: 'Wen', pence: 280 });
});
