import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough, Writable } from 'node:stream';
import { EventLoop, TerminalSessionStore, createNatlangRuntime, loadNatlang, openAICompatibleModelTurn,
  renderTerminalView, runTerminalShell } from '../dist/index.js';
import { CommandRecipeLibrary } from '../../applications/dist/terminal/index.js';
import { readEvidencePath, STARTER_EVIDENCE } from '../../applications/dist/evidence/index.js';

test('terminal shell exposes built-in and application discovery commands', async () => {
  let text = '';
  const output = new Writable({ write(chunk, _encoding, done) { text += String(chunk); done(); } });
  output.isTTY = false; output.columns = 80;
  const view = { title: 'Guided app', blocks: [], prompt: 'guide> ', help: ['/sources collection'] };
  const app = { view, async start() { return { view }; }, async refresh() { return { view }; },
    async consume(events) { for await (const _event of events) {} }, cancel() {}, async close() {} };
  const input = new PassThrough();
  setTimeout(() => input.write('/help\n'), 5);
  setTimeout(() => input.write('/sources\n'), 15);
  setTimeout(() => input.end('/quit\n'), 25);
  await runTerminalShell(app, { input, output,
    event: (value, id) => ({ id, value }), commands: {
      sources: { description: 'list loaded sources', run: () => 'welcome\nworkflow' },
    } });
  assert.match(text, /\/help show commands/);
  assert.match(text, /\/sources list loaded sources/);
  assert.match(text, /welcome\nworkflow/);
});

test('terminal shell keeps piped lines that arrive during a step and returns at end of input', async () => {
  const output = new Writable({ write(_chunk, _encoding, done) { done(); } });
  output.isTTY = false; output.columns = 80;
  const view = { title: 'Piped', blocks: [] }, seen = [];
  const app = { view, async start() { return { view }; }, async refresh() { return { view }; },
    async consume(events) { for await (const event of events) { seen.push(event.value); await new Promise(done => setTimeout(done, 20)); } },
    cancel() {}, async close() {} };
  const input = new PassThrough();
  input.end('first\nsecond\n');
  await runTerminalShell(app, { input, output, event: (value, id) => ({ id, value }) });
  assert.deepEqual(seen, ['first', 'second']);
});

test('evidence console has starter material and can discover a source directory', () => {
  assert.ok(STARTER_EVIDENCE.length >= 3);
  const folder = mkdtempSync(join(tmpdir(), 'natlang-evidence-onboarding-'));
  try {
    writeFileSync(join(folder, 'guide.md'), 'A useful guide.');
    writeFileSync(join(folder, 'package.json'), '{"name":"not-evidence"}');
    const documents = readEvidencePath('.', folder);
    assert.deepEqual(documents.map(row => row.id), ['guide']);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a terminal session store persists committed state and restores duplicate suppression', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-terminal-app-'));
  const store = new TerminalSessionStore(join(folder, 'session.json'));
  const make = checkpoint => {
    const loop = new EventLoop({ initialState: checkpoint.state, initialRevision: checkpoint.revision,
      seenEventIds: checkpoint.seen_event_ids, reduce: state => ({ count: state.count + 1 }),
      view: state => ({ blocks: [{ kind: 'text', text: `Count ${state.count}` }] }),
      onCommit: commit => store.commit(commit, loop.seenEventIds) });
    return loop;
  };
  const loop = make(store.load({ count: 0 }));
  try {
    assert.match((await loop.start()).view.blocks[0].text, /0/);
    const results = await Promise.all([loop.dispatch({ id: 'one', kind: 'add' }), loop.dispatch({ id: 'two', kind: 'add' })]);
    assert.deepEqual(results.map(result => result.revision), [1, 2]);
    const saved = store.load({ count: -1 });
    assert.equal(saved.revision, 2); assert.equal(saved.state.count, 2);
    assert.deepEqual(saved.seen_event_ids, ['one', 'two']);
    assert.equal(readFileSync(`${store.path}.events.jsonl`, 'utf8').trim().split('\n').length, 2);
    const restored = make(saved);
    await restored.start();
    assert.equal(await restored.dispatch({ id: 'one', kind: 'add' }), null);
    assert.equal(restored.state.count, 2);
    await restored.close();
  } finally { await loop.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('event consumption reports a failed reduction and continues with later events', async () => {
  let reductions = 0;
  const loop = new EventLoop({ initialState: { count: 0 }, view: () => ({ blocks: [] }), reduce: () => {
    if (reductions++ === 0) throw new Error('malformed model turn');
    return { count: 1 };
  } });
  const failures = [], transitions = [];
  async function* events() { yield { id: 'bad', kind: 'request' }; yield { id: 'good', kind: 'request' }; }
  try {
    await loop.start();
    await loop.consume(events(), transition => transitions.push(transition), (error, event) => failures.push({ error: String(error), id: event.id }));
    assert.deepEqual(failures, [{ error: 'Error: malformed model turn', id: 'bad' }]);
    assert.equal(transitions.length, 1);
    assert.equal(loop.state.count, 1);
  } finally { await loop.close(); }
});

test('a failed model transport still delivers the invocation trace', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-failed-trace-'));
  writeFileSync(join(folder, 'main.nl'), '---\nargs: {}\nreturns: string\n---\nReturn a short greeting.\n');
  const traces = [];
  const runtime = createNatlangRuntime({ trace: trace => traces.push(trace),
    model: () => { throw new SyntaxError('malformed tool arguments'); } });
  try {
    await assert.rejects(runtime.run(() => loadNatlang(join(folder, 'main.nl'))()), /malformed tool arguments/);
    assert.equal(traces.length, 1);
    assert.ok(traces[0].events.some(event => event.kind === 'model_request' && event.phase === 'error' &&
      String(event.error).includes('malformed tool arguments')));
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('terminal renderer checks tables and strips view-supplied control characters', () => {
  const output = renderTerminalView({ title: '\u001b[31mUnsafe', blocks: [
    { kind: 'table', columns: ['Name', 'State'], rows: [['task', 'ready']] },
    { kind: 'status', text: 'ok', tone: 'good' },
  ] }, { width: 40, color: false });
  assert.doesNotMatch(output, /\u001b/); assert.match(output, /Name\s+\| State/); assert.match(output, /task/);
  assert.throws(() => renderTerminalView({ blocks: [
    { kind: 'table', columns: ['a'], rows: [['x', 'y']] },
  ] }), /match columns/);
});

test('OpenAI-compatible driver sends and returns the current tool name unchanged', async () => {
  const original = globalThis.fetch; let wire;
  globalThis.fetch = async (_url, init) => {
    wire = JSON.parse(init.body);
    return new Response(JSON.stringify({ choices: [{ message: { content: '', tool_calls: [
      { function: { name: 'eval', arguments: '{"code":"work()"}' } },
    ] } }], usage: { completion_tokens: 4, prompt_tokens: 9 } }), { status: 200,
      headers: { 'content-type': 'application/json' } });
  };
  try {
    const driver = openAICompatibleModelTurn({ endpoint: 'http://model.test', model: 'fixture' });
    const result = await driver({ messages: [], tools: [{ type: 'function', function: {
      name: 'eval', parameters: { type: 'object' } } }], temperature: 0, seed: 3, max_tokens: null });
    assert.equal(wire.tools[0].function.name, 'eval');
    assert.equal('max_tokens' in wire, false);
    assert.deepEqual(result.calls, [['eval', { code: 'work()' }]]);
    assert.equal(result.prompt_tokens, 9);
  } finally { globalThis.fetch = original; }
});

test('OpenAI-compatible driver retries one malformed tool call with corrective context', async () => {
  const original = globalThis.fetch, wires = [];
  globalThis.fetch = async (_url, init) => {
    wires.push(JSON.parse(init.body));
    const malformed = wires.length === 1;
    return new Response(JSON.stringify({ choices: [{ finish_reason: 'tool_calls', message: {
      content: '', tool_calls: [{ function: { name: 'eval',
        arguments: malformed ? '{"code":"ok"' : '{"code":"ok"}' } }],
    } }], usage: { completion_tokens: 2, prompt_tokens: 5 } }), { status: 200,
      headers: { 'content-type': 'application/json' } });
  };
  try {
    const driver = openAICompatibleModelTurn({ endpoint: 'http://model.test', model: 'fixture' });
    const result = await driver({ messages: [{ role: 'user', content: 'Return ok.' }],
      tools: [{ type: 'function', function: { name: 'eval', parameters: { type: 'object' } } }],
      temperature: 0, seed: 3, max_tokens: null });
    assert.equal(wires.length, 2);
    assert.match(wires[1].messages.at(-1).content, /valid JSON object arguments/);
    assert.deepEqual(result.calls, [['eval', { code: 'ok' }]]);
    assert.equal(result.prompt_tokens, 10);
    assert.equal(result.completion_tokens, 4);
  } finally { globalThis.fetch = original; }
});

test('OpenAI-compatible driver bounds malformed-call repair', async () => {
  const original = globalThis.fetch; let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    return new Response(JSON.stringify({ choices: [{ finish_reason: 'tool_calls', message: {
      content: '', tool_calls: [{ function: { name: 'eval', arguments: '{"code":' } }],
    } }] }), { status: 200, headers: { 'content-type': 'application/json' } });
  };
  try {
    const driver = openAICompatibleModelTurn({ endpoint: 'http://model.test', model: 'fixture' });
    await assert.rejects(driver({ messages: [], tools: [], temperature: 0, seed: 3, max_tokens: null }),
      /malformed tool arguments after 2 attempts/);
    assert.equal(requests, 2);
  } finally { globalThis.fetch = original; }
});

test('command recipes use argv, contain cwd, bound output, and propagate abort', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-recipes-'));
  try {
    assert.throws(() => new CommandRecipeLibrary(folder, [{ id: 'escape', description: 'escape',
      cwd: '..', argv: [process.execPath, '-e', ''] }]), /escapes workspace/);
    const library = new CommandRecipeLibrary(folder, [
      { id: 'print', description: 'print bounded output',
        argv: [process.execPath, '-e', 'process.stdout.write("123456789")'] },
      { id: 'wait', description: 'wait until cancelled',
        argv: [process.execPath, '-e', 'setTimeout(()=>{},60000)'] },
    ], { outputBytes: 5 });
    const printed = await library.recipes()[0].run({});
    assert.equal(printed.status, 'ok'); assert.match(printed.detail, /56789/);
    assert.doesNotMatch(printed.detail, /12345/);
    const controller = new AbortController();
    const waiting = library.recipes()[1].run({ signal: controller.signal });
    controller.abort();
    const cancelled = await waiting;
    assert.equal(cancelled.status, 'unknown'); assert.match(cancelled.detail, /Cancellation requested/);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
