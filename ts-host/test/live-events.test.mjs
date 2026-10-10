/**
 * Live events for natlang functions (plans/STREAMING.md §1.6): the runtime passes each call's model deltas to its
 * live subscribers, tagged with the call and turn; nested calls are told apart; a re-sent request's `reset` reaches
 * them; without a subscriber the driver is called exactly as before and nothing else changes.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { openAICompatibleModelTurn } from '../dist/model/openai-compatible.js';
import { scriptedModel } from './support/natlang.mjs';

const chunk = (delta, finish = null) => ({ id: 'c1', object: 'chat.completion.chunk', created: 1, model: 'm',
  choices: [{ index: 0, delta, finish_reason: finish }] });
/** Whether the model's eval has run (the runtime's scope_ pre-fills are not the model's; a malformed-call retry is not). */
const evaluated = messages => messages.some(message => message.role === 'tool' && !String(message.tool_call_id).startsWith('scope_'));

/**
 * A model server that streams: a call's first turn says `Working on NAME.` and calls eval with the function's code;
 * later turns say `done`. `malformedOnce` names a function whose first reply carries unparsable call arguments once.
 */
async function server({ malformedOnce } = {}) {
  const bodies = [];
  let malformed = false;
  const instance = createServer((incoming, response) => {
    let text = '';
    incoming.setEncoding('utf8'); incoming.on('data', part => { text += part; });
    incoming.on('end', () => {
      // Only chat completions are served (the driver's context-window probe gets a 404).
      if (incoming.url !== '/v1/chat/completions') { response.writeHead(404); response.end(); return; }
      const body = JSON.parse(text); bodies.push(body);
      const opening = String(body.messages[1]?.content ?? '');
      const name = opening.includes('Double n') ? 'addOne' : 'main';
      const code = name === 'addOne' ? 'return 2 * n + 1;' : 'return await addOne(n);';
      let chunks;
      if (evaluated(body.messages)) chunks = [chunk({ role: 'assistant', content: 'do' }), chunk({ content: 'ne' }, 'stop')];
      else {
        const args = JSON.stringify({ code });
        const bad = name === malformedOnce && !malformed;
        if (bad) malformed = true;
        chunks = [chunk({ role: 'assistant', content: `Working on ` }), chunk({ content: `${name}.` }),
          chunk({ tool_calls: [{ index: 0, id: `call_${name}`, type: 'function', function: { name: 'eval',
            arguments: bad ? '{bad' : args.slice(0, 5) } }] }),
          ...(bad ? [] : [chunk({ tool_calls: [{ index: 0, function: { arguments: args.slice(5) } }] })]),
          chunk({}, 'tool_calls')];
      }
      if (!body.stream) throw new Error('the test server only streams');
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(chunks.map(item => `data: ${JSON.stringify(item)}\n\n`).join('') + 'data: [DONE]\n\n');
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${instance.address().port}`, bodies,
    close: () => new Promise(resolve => instance.close(resolve)) };
}

/** main(n) calls its companion function addOne(n) = 2n + 1. */
function program() {
  const root = mkdtempSync(join(tmpdir(), 'live-events-'));
  const write = (path, text) => { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); };
  write('natlang.json', '{}');
  write('main.nl', '---\nargs:\n  n: number\nreturns: number\n---\nAdd one to n with addOne.\n');
  write('main/addOne.nl', '---\nargs:\n  n: number\nreturns: number\n---\nDouble n, then add one.\n');
  return loadNatlang(join(root, 'main.nl'));
}

/** The text a subscriber shows per call and turn: text deltas appended, a reset discarding what came before. */
function shown(events) {
  const views = new Map();
  for (const event of events) {
    const key = `${event.name}#${event.turn}`;
    if (event.delta.type === 'reset') views.set(key, '');
    else if (event.delta.type === 'text') views.set(key, (views.get(key) ?? '') + event.delta.text);
  }
  return Object.fromEntries(views);
}

test('a natlang call\'s deltas reach the subscriber in order, tagged with the call; nested calls are told apart', async () => {
  const model = await server();
  try {
    const events = [];
    const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: model.endpoint, model: 'm' }),
      calls: false, onLive: event => events.push(event) });
    const main = program();
    const traces = [];
    assert.equal(await runtime.run(() => main(5), { trace: trace => traces.push(trace) }), 11);
    assert.ok(events.every(event => event.type === 'model_delta'));
    const outer = events.filter(event => event.name === 'main'), inner = events.filter(event => event.name === 'addOne');
    assert.equal(new Set(outer.map(event => event.callId)).size, 1);
    assert.equal(new Set(inner.map(event => event.callId)).size, 1);
    assert.notEqual(outer[0].callId, inner[0].callId);
    assert.equal(outer[0].parentCallId, null);
    assert.equal(inner[0].parentCallId, outer[0].callId, 'the nested call names its caller');
    assert.equal(new Set(events.map(event => event.taskId)).size, 1);
    // The outer call's first turn streams, then the inner call runs inside its eval, then the outer call finishes.
    assert.deepEqual(events.map(event => `${event.name}#${event.turn}:${event.delta.type}`), [
      'main#1:text', 'main#1:text', 'main#1:tool_call', 'main#1:tool_call',
      'addOne#1:text', 'addOne#1:text', 'addOne#1:tool_call', 'addOne#1:tool_call',
      'addOne#2:text', 'addOne#2:text', 'main#2:text', 'main#2:text']);
    assert.deepEqual(shown(events), { 'main#1': 'Working on main.', 'addOne#1': 'Working on addOne.',
      'addOne#2': 'done', 'main#2': 'done' });
    const calls = events.filter(event => event.delta.type === 'tool_call' && event.name === 'addOne').map(event => event.delta);
    assert.equal(calls[0].name, 'eval');
    assert.deepEqual(JSON.parse(calls.map(delta => delta.arguments).join('')), { code: 'return 2 * n + 1;' });
    // The call identity is the one the trace records; the turn is the one its model_request events give.
    const traced = Object.fromEntries(traces.map(trace => [trace.name, trace]));
    assert.equal(traced.main.callId, outer[0].callId);
    assert.equal(traced.addOne.callId, inner[0].callId);
    assert.equal(outer[0].definitionId, traced.main.definitionId);
    assert.deepEqual([...new Set(traced.addOne.events.filter(event => event.kind === 'model_request').map(event => event.turn))], [1, 2]);
  } finally { await model.close(); }
});

test('a re-sent request\'s reset reaches the subscriber, so the abandoned attempt\'s output is discarded', async () => {
  const model = await server({ malformedOnce: 'addOne' });
  try {
    const events = [];
    const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: model.endpoint, model: 'm' }), calls: false });
    const main = program();
    // A task's own subscriber.
    assert.equal(await runtime.run(() => main(2), { onLive: event => events.push(event) }), 5);
    const inner = events.filter(event => event.name === 'addOne' && event.turn === 1).map(event => event.delta.type);
    assert.deepEqual(inner, ['text', 'text', 'tool_call', 'reset', 'text', 'text', 'tool_call', 'tool_call']);
    assert.equal(shown(events)['addOne#1'], 'Working on addOne.');
  } finally { await model.close(); }
});

test('a subscriber that throws does not affect the call; a runtime and a task subscriber both receive', async () => {
  const model = await server();
  try {
    const a = [], b = [];
    const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: model.endpoint, model: 'm' }),
      calls: false, onLive: event => { a.push(event); throw new Error('observer'); } });
    assert.equal(await runtime.run(() => program()(1), { onLive: event => b.push(event) }), 3);
    assert.equal(a.length, 12);
    assert.deepEqual(a, b);
  } finally { await model.close(); }
});

test('without a subscriber the driver gets no turn options, and requests and results are unchanged', async () => {
  // A driver called with two arguments when nobody listens, and with an onDelta option when someone does.
  const argumentCounts = [];
  const scripted = scriptedModel(opening => opening.includes('Double n') ? 'return 2 * n + 1;' : 'return await addOne(n);');
  const driver = (...args) => { argumentCounts.push(args.length); return scripted.driver(args[0]); };
  assert.equal(await createNatlangRuntime({ model: driver, calls: false }).run(() => program()(4)), 9);
  assert.ok(argumentCounts.length >= 4 && argumentCounts.every(count => count === 2), String(argumentCounts));
  argumentCounts.length = 0;
  const events = [];
  assert.equal(await createNatlangRuntime({ model: driver, calls: false, onLive: event => events.push(event) }).run(() => program()(4)), 9);
  assert.ok(argumentCounts.every(count => count === 3), String(argumentCounts));
  assert.deepEqual(events, [], 'a driver that does not stream emits no deltas');

  // Over a streaming server: the same requests and result with and without a subscriber.
  const normalize = bodies => JSON.parse(JSON.stringify(bodies).replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, 'UUID'));
  const runs = [], main = program();
  for (const onLive of [undefined, () => {}]) {
    const model = await server();
    try {
      const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: model.endpoint, model: 'm' }),
        calls: false, ...(onLive ? { onLive } : {}) });
      runs.push({ value: await runtime.run(() => main(3)), bodies: normalize(model.bodies) });
    } finally { await model.close(); }
  }
  assert.equal(runs[0].value, 7);
  assert.deepEqual(runs[1], runs[0]);
});
