import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PassThrough, Writable } from 'node:stream';
import { NatlangHost, TerminalEventQueue, TerminalNatlangApplication, TerminalSessionStore,
  NodeFileTree, TypeScriptEnvironment, openAICompatibleModelTurn, renderTerminalView, runTerminalShell } from '../dist/index.js';
import { RecipeTerminal } from '../../applications/semantic_terminal.mjs';
import { NotebookWorkspace } from '../../applications/notebook.mjs';
import { CommandRecipeLibrary } from '../../applications/terminal_recipes.mjs';
import { readEvidencePath, STARTER_EVIDENCE } from '../../applications/package_targets/evidence_console.mjs';

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

test('terminal application serializes events, persists state, and restores duplicate suppression', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-terminal-app-'));
  const types = join(folder, 'types.ts'), reducer = join(folder, 'reduce.ts'), view = join(folder, 'view.ts');
  writeFileSync(types, 'export type State = { count: number };\nexport type Event = { id: string, kind: string };\nexport type Block = { kind: string, text: string };\nexport type View = { blocks: Block[] };');
  writeFileSync(reducer, 'export default function reduce(state: State, event: Event): State {\n  return {count: state.count + 1};\n}');
  writeFileSync(view, 'export default function view(state: State): View {\n  return {blocks:[{kind:"text",text:`Count ${state.count}`}]};\n}');
  const store = new TerminalSessionStore(join(folder, 'session.json'));
  const host = new NatlangHost(); let app;
  app = new TerminalNatlangApplication({ runner: host, source: { reducer, view }, initialState: { count: 0 },
    onCommit: commit => store.commit(commit, app.seenEventIds) });
  try {
    assert.match((await app.start()).view.blocks[0].text, /0/);
    const results = await Promise.all([app.dispatch({ id: 'one', kind: 'add' }),
      app.dispatch({ id: 'two', kind: 'add' })]);
    assert.deepEqual(results.map(result => result.revision), [1, 2]);
    assert.equal(app.state.count, 2);
    assert.equal(await app.dispatch({ id: 'one', kind: 'add' }), null);
    const saved = store.load({ count: -1 });
    assert.equal(saved.revision, 2); assert.equal(saved.state.count, 2);
    assert.deepEqual(saved.seen_event_ids, ['one', 'two']);
    assert.equal(readFileSync(`${store.path}.events.jsonl`, 'utf8').trim().split('\n').length, 2);
  } finally { await app.close(); host.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('terminal event consumption reports a failed reduction and continues with later events', async () => {
  let reductions = 0;
  const runner = { async run(request) {
    if (request.source.path === 'reduce') {
      if (reductions++ === 0) throw new Error('malformed model turn');
      return { outcome: { kind: 'done' }, value: { count: 1 } };
    }
    return { outcome: { kind: 'done' }, value: { blocks: [] } };
  } };
  const app = new TerminalNatlangApplication({ runner,
    source: { reducer: 'reduce', view: 'view' }, initialState: { count: 0 } });
  const failures = [], transitions = [];
  async function* events() {
    yield { id: 'bad', kind: 'request' };
    yield { id: 'good', kind: 'request' };
  }
  try {
    await app.start();
    await app.consume(events(), transition => transitions.push(transition),
      (error, event) => failures.push({ error: String(error), id: event.id }));
    assert.deepEqual(failures, [{ error: 'Error: reduce failed: Error: malformed model turn', id: 'bad' }]);
    assert.equal(transitions.length, 1);
    assert.equal(app.state.count, 1);
  } finally { await app.close(); }
});

test('native host preserves a trace when the model transport fails', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-failed-trace-'));
  const source = join(folder, 'main.nl'), tracePath = join(folder, 'failed.jsonl');
  writeFileSync(source, '---\nargs: {}\nreturns: string\n---\nReturn a short greeting.');
  const host = new NatlangHost();
  try {
    await assert.rejects(host.run({ source: { kind: 'file', path: source }, tracePath,
      modelTurn: () => { throw new SyntaxError('malformed tool arguments'); } }), /malformed tool arguments/);
    const events = readFileSync(tracePath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.ok(events.some(event => event.kind === 'model_request' && event.phase === 'error' &&
      String(event.error).includes('malformed tool arguments')));
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
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

test('notebook can import cells and tables after startup without a fixture restart', async () => {
  const notebook = new NotebookWorkspace([], {},
    { environment: new TypeScriptEnvironment({ mode: 'fresh' }) });
  try {
    const loaded = notebook.importConfig({ tables: { measurements: [{ amount: 2 }, { amount: 5 }] }, cells: [
      { id: 'total', engine: 'sqlite', needs: [], description: 'sum values',
        source: 'SELECT SUM(amount) AS total FROM measurements' },
    ] });
    assert.deepEqual(loaded.tables, ['measurements']);
    assert.equal(notebook.catalog()[0].id, 'total');
    assert.match((await notebook.execute('total')).sample, /7/);
    notebook.importConfig({ tables: { measurements: [{ amount: 11 }] }, cells: [] });
    assert.match((await notebook.execute('total')).sample, /11/);
  } finally { notebook.close(); }
});
