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
  writeFileSync(types, 'export type State = { count: Num };\nexport type Event = { id: Text, kind: Text };\nexport type Block = { kind: Text, text: Text };\nexport type View = { blocks: Block[] };');
  writeFileSync(reducer, '/*---\nengine: typescript-host\nargs:\n  state: State\n  event: Event\nreturns: State\n---*/\nreturn {count: args.state.count + 1};');
  writeFileSync(view, '/*---\nengine: typescript-host\nargs:\n  state: State\nreturns: View\n---*/\nreturn {blocks:[{kind:"text",text:`Count ${args.state.count}`}]};');
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

test('terminal reducer gets a fresh lazy file view while crisp view stays file-free', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-terminal-files-'));
  const types = join(folder, 'types.ts'), reducer = join(folder, 'reduce.nl'), view = join(folder, 'view.ts');
  writeFileSync(join(folder, 'note.txt'), 'first note');
  writeFileSync(types, 'export type State = { count: Num, note: Text }; export type Event = { id: Text, kind: Text }; export type View = { text: Text }; export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };');
  writeFileSync(reducer, '---\nargs:\n  state: State\n  event: Event\n  files: Dict<File>\nreturns: State\n---\nRead args/files/note.txt/text and retain it while incrementing the count.');
  writeFileSync(view, '/*---\nengine: typescript-host\nargs:\n  state: State\nreturns: View\n---*/\nreturn { text: `${args.state.count}:${args.state.note}` };');
  const host = new NatlangHost(), seen = [];
  let turn = 0;
  const app = new TerminalNatlangApplication({ runner: host, source: { reducer, view },
    initialState: { count: 0, note: '' }, reducerInputs: () => ({ files: new NodeFileTree(folder) }),
    modelTurn: request => {
      seen.push(JSON.stringify(request.messages));
      turn++;
      if (String(request.messages.at(-1)?.content ?? '').includes('return: complete'))
        return { calls: [], text: 'done', completion_tokens: 1 };
      if (turn % 2 === 0) return { calls: [['write', { path: 'return', value:
        turn === 2 ? { count: 1, note: 'first note' } : { count: 2, note: 'second note' }, done: 1 }]], completion_tokens: 1 };
      return { calls: [['read', { path: 'args/files/note.txt/text' }]], completion_tokens: 1 };
    } });
  try {
    assert.equal((await app.start()).view.text, '0:');
    assert.equal((await app.dispatch({ id: 'one', kind: 'request' })).view.text, '1:first note');
    writeFileSync(join(folder, 'note.txt'), 'second note');
    assert.equal((await app.dispatch({ id: 'two', kind: 'request' })).view.text, '2:second note');
    assert.ok(seen.some(message => message.includes('second note')));
  } finally { await app.close(); host.close(); rmSync(folder, { recursive: true, force: true }); }
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

test('OpenAI-compatible driver aliases tools without changing natlang call names', async () => {
  const original = globalThis.fetch; let wire;
  globalThis.fetch = async (_url, init) => {
    wire = JSON.parse(init.body);
    return new Response(JSON.stringify({ choices: [{ message: { content: '', tool_calls: [
      { function: { name: 'call_function', arguments: '{"function":"work","to":"return"}' } },
    ] } }], usage: { completion_tokens: 4, prompt_tokens: 9 } }), { status: 200,
      headers: { 'content-type': 'application/json' } });
  };
  try {
    const driver = openAICompatibleModelTurn({ endpoint: 'http://model.test', model: 'fixture',
      toolAliases: { call: 'call_function' } });
    const result = await driver({ messages: [], tools: [{ type: 'function', function: {
      name: 'call', parameters: { type: 'object' } } }], temperature: 0, seed: 3, max_tokens: null });
    assert.equal(wire.tools[0].function.name, 'call_function');
    assert.equal('max_tokens' in wire, false);
    assert.deepEqual(result.calls, [['call', { function: 'work', to: 'return' }]]);
    assert.equal(result.prompt_tokens, 9);
  } finally { globalThis.fetch = original; }
});

test('OpenAI-compatible driver retries one malformed tool call with corrective context', async () => {
  const original = globalThis.fetch, wires = [];
  globalThis.fetch = async (_url, init) => {
    wires.push(JSON.parse(init.body));
    const malformed = wires.length === 1;
    return new Response(JSON.stringify({ choices: [{ finish_reason: 'tool_calls', message: {
      content: '', tool_calls: [{ function: { name: 'write',
        arguments: malformed ? '{"path":"return"' : '{"path":"return","value":"ok"}' } }],
    } }], usage: { completion_tokens: 2, prompt_tokens: 5 } }), { status: 200,
      headers: { 'content-type': 'application/json' } });
  };
  try {
    const driver = openAICompatibleModelTurn({ endpoint: 'http://model.test', model: 'fixture' });
    const result = await driver({ messages: [{ role: 'user', content: 'Return ok.' }],
      tools: [{ type: 'function', function: { name: 'write', parameters: { type: 'object' } } }],
      temperature: 0, seed: 3, max_tokens: null });
    assert.equal(wires.length, 2);
    assert.match(wires[1].messages.at(-1).content, /valid JSON object arguments/);
    assert.deepEqual(result.calls, [['write', { path: 'return', value: 'ok' }]]);
    assert.equal(result.prompt_tokens, 10);
    assert.equal(result.completion_tokens, 4);
  } finally { globalThis.fetch = original; }
});

test('OpenAI-compatible driver bounds malformed-call repair', async () => {
  const original = globalThis.fetch; let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    return new Response(JSON.stringify({ choices: [{ finish_reason: 'tool_calls', message: {
      content: '', tool_calls: [{ function: { name: 'write', arguments: '{"path":' } }],
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

test('semantic terminal receives native job completion through the shared event queue', { timeout: 5000 }, async () => {
  const terminal = new RecipeTerminal([{ id: 'inspect', description: 'inspect the workspace',
    run: async () => ({ status: 'ok', detail: 'clean' }) }]);
  const queue = new TerminalEventQueue();
  terminal.subscribe(event => { queue.push(event); queue.close(); });
  const host = new NatlangHost({ host: { terminal, drainEvents: () => terminal.drainEvents() }, mode: 'retained' });
  const source = name => fileURLToPath(new URL(`../../codebases/semantic_terminal/${name}`, import.meta.url));
  const initialState = { revision: 0, active_request: '', active_job: '', status: 'idle', messages: [], history: [] };
  const seenPrompts = []; let stepCount = 0;
  const driver = turn => {
    if (turn.messages.filter(message => message.role === 'assistant').length > 1)
      return { calls: [], text: 'done', completion_tokens: 1 };
    const prompt = String(turn.messages.find(message => message.role === 'user')?.content ?? '');
    seenPrompts.push(prompt.slice(0, 500));
    if (prompt.includes('function reduce(')) return { calls: [['call', { function: 'step', to: 'return',
      inputs: { acc: 'args/state', item: 'args/event', files: 'args/files' } }]], completion_tokens: 1 };
    if (prompt.includes('function step(') && stepCount++ === 0) return { calls: [
      ['call', { function: 'recipes', to: 'let/catalog' }],
      ['call', { function: 'interpret', to: 'let/recipe', inputs: {
        text: 'args/item/text', catalog: 'let/catalog', files: 'args/files' } }],
      ['call', { function: 'launch', to: 'return', inputs: { acc: 'args/acc', item: 'args/item', recipe: 'let/recipe' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('function step(')) return { calls: [
      ['call', { function: 'explain', to: 'let/message', inputs: { item: 'args/item' } }],
      ['call', { function: 'settle', to: 'return', inputs: { acc: 'args/acc', item: 'args/item', message: 'let/message' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('Choose one recipe ID')) return { calls: [['write', { path: 'return', value: 'inspect' }]], completion_tokens: 1 };
    return { calls: [['write', { path: 'return', value: 'Inspection completed: clean.' }]], completion_tokens: 1 };
  };
  const app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    reducerInputs: () => ({ files: new NodeFileTree(process.cwd()) }), initialState, modelTurn: driver });
  try {
    await app.start();
    await app.dispatch({ kind: 'request', id: 'r1', request_id: '', job_id: '', text: 'inspect', status: '', detail: '' });
    assert.equal(terminal.jobs.size, 1, JSON.stringify(seenPrompts));
    await app.consume(queue);
    assert.equal(app.state.status, 'ok'); assert.equal(app.state.history.length, 1);
    assert.match(app.state.messages.at(-1), /completed/);
  } finally { await app.close(); host.close(); }
});

test('notebook console keeps semantic goal selection and dependency traversal in natlang', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'facts', engine: 'sqlite', needs: [], description: 'sum the values',
      source: 'SELECT SUM(value) AS total FROM numbers' },
    { id: 'report', engine: 'typescript-host', needs: ['facts'], description: 'final total report',
      source: 'return {total: args.deps.facts[0].total};' },
  ], { numbers: [{ value: 2 }, { value: 5 }] },
  { environment: new TypeScriptEnvironment({ mode: 'fresh' }) });
  const host = new NatlangHost({ host: { notebook, drainEvents: () => notebook.drainEvents() }, mode: 'retained' });
  const source = name => fileURLToPath(new URL(`../../codebases/notebook_console/${name}`, import.meta.url));
  let choice = 0;
  const driver = turn => {
    if (turn.messages.filter(message => message.role === 'assistant').length > 1)
      return { calls: [], text: 'done', completion_tokens: 1 };
    const prompt = String(turn.messages.find(message => message.role === 'user')?.content ?? '');
    if (prompt.includes('function reduce(')) return { calls: [
      ['call', { function: 'catalog', to: 'let/cells' }],
      ['call', { function: 'choose_goal', to: 'let/goal', inputs: { request: 'args/event/value', cells: 'let/cells' } }],
      ['call', { function: 'run_notebook', to: 'let/result', inputs: {
        goal: 'let/goal', question: 'args/event/value', files: 'args/files' } }],
      ['call', { function: 'append', to: 'return', inputs: { state: 'args/state', request: 'args/event/value', result: 'let/result' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('Choose exactly one offered cell ID'))
      return { calls: [['write', { path: 'return', value: 'report' }]], completion_tokens: 1 };
    if (prompt.includes('function run(')) return { calls: [
      ['call', { function: 'prepare', to: 'let/initial', inputs: { goal: 'args/goal' } }],
      ['call', { function: 'step', to: 'let/finished', until: 'complete', init: 'let/initial', max: 2,
        inputs: { files: 'args/files' } }],
      ['call', { function: 'explain', to: 'let/answer', inputs: {
        question: 'args/question', state: 'let/finished', files: 'args/files' } }],
      ['call', { function: 'attach', to: 'return', inputs: { state: 'let/finished', answer: 'let/answer' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('function step(')) return { calls: [
      ['call', { function: 'ready_cells', to: 'let/ready', inputs: { state: 'args/state' } }],
      ['call', { function: 'choose', to: 'let/chosen', inputs: {
        ready: 'let/ready', goal: 'args/state/goal', files: 'args/files' } }],
      ['call', { function: 'advance', to: 'return', inputs: { state: 'args/state', chosen: 'let/chosen' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('Choose one offered ready cell'))
      return { calls: [['write', { path: 'return', value: choice++ === 0 ? 'facts' : 'report' }]], completion_tokens: 1 };
    return { calls: [['write', { path: 'return', value: 'The report cell gives total 7.' }]], completion_tokens: 1 };
  };
  const app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    reducerInputs: () => ({ files: new NodeFileTree(process.cwd()) }),
    initialState: { requests: [], runs: [], status: 'idle' }, modelTurn: driver });
  try {
    await app.start();
    await app.dispatch({ id: 'q1', kind: 'request', value: 'What is the final total?' });
    assert.equal(app.state.status, 'done');
    assert.deepEqual(Array.from(app.state.runs[0].order), ['facts', 'report']);
    assert.match(app.state.runs[0].answer, /total 7/);
  } finally { await app.close(); host.close(); notebook.close(); }
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
