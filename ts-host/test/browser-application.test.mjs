import assert from 'node:assert/strict';
import { test } from 'node:test';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

const files = {
  'types.ts': 'export type Counter = { count: number };\nexport type UiEvent = { id: string, kind: string, value?: string };\nexport type View = { tag: string, text: string };',
  'reduce.ts': 'export default function reduce(state: Counter, event: UiEvent): Counter {\n  if (event.kind === "bad") return { count: "wrong" };\n  return { count: state.count + 1 };\n}',
  'view.ts': 'export default function view(state: Counter): View {\n  return { tag: "p", text: `Count ${state.count}` };\n}',
};

test('browser application serializes incoming events and publishes each reduced view', async () => {
  const { BrowserNatlangClient, BrowserNatlangApplication } = await api();
  const client = new BrowserNatlangClient();
  const transitions = [], failures = [];
  const app = new BrowserNatlangApplication({ client,
    source: { files, reducer: 'reduce.ts', view: 'view.ts' },
    initialState: { count: 0 }, seedRoot: 42,
    onTransition: item => transitions.push(item), onFailure: item => failures.push(item) });
  try {
    assert.equal((await app.start()).view.text, 'Count 0');
    const pending = [1, 2, 3].map(n => app.dispatch({ id: `e${n}`, kind: 'increment' }));
    const finished = await Promise.all(pending);
    assert.deepEqual(finished.map(item => item.revision), [1, 2, 3]);
    assert.equal(app.state.count, 3);
    assert.equal(app.view.text, 'Count 3');
    assert.equal(await app.dispatch({ id: 'e2', kind: 'increment' }), null);
    await assert.rejects(app.dispatch({ id: 'bad', kind: 'bad' }), /reduce failed/);
    assert.equal(app.state.count, 3);
    assert.equal(failures[0].stage, 'reduce');
    assert.equal((await app.dispatch({ id: 'e4', kind: 'increment' })).revision, 4);
    assert.equal(transitions.length, 5);
    assert.ok(transitions[1].reducerRun.trace.length > 0);
  } finally { await app.close(); await client.close(); }
  await assert.rejects(app.dispatch({ id: 'late', kind: 'increment' }), /closed/);
});

test('natlang can generate the view while the app contract remains independent of the renderer', async () => {
  const { BrowserNatlangClient, BrowserNatlangApplication } = await api();
  const client = new BrowserNatlangClient();
  const naturalFiles = { ...files, 'view.nl': `---\nargs:\n  state: Counter\nreturns: View\n---\nDescribe the current count as a short paragraph UI node.` };
  const app = new BrowserNatlangApplication({ client,
    source: { files: naturalFiles, reducer: 'reduce.ts', view: 'view.nl' },
    initialState: { count: 0 }, modelTurn: turn => {
      if (turn.messages.filter(message => message.role === 'assistant').length > 0)
        return { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
      return { calls: [['eval', { code: '({ tag: "p", text: "Natlang view " + state.count })' }]], completion_tokens: 1 };
    } });
  try {
    assert.equal((await app.start()).view.text, 'Natlang view 0');
    assert.equal((await app.dispatch({ id: 'one', kind: 'increment' })).view.text,
      'Natlang view 1');
    assert.equal(app.state.count, 1);
  } finally { await app.close(); await client.close(); }
});

test('browser semantic reducer receives fresh file inputs while crisp view remains portable', async () => {
  const api = await (async () => {
    const process = globalThis.process;
    try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
    finally { globalThis.process = process; }
  })();
  const project = {
    'types.ts': 'export type Counter = { count: number, note: string }; export type UiEvent = { id: string, kind: string }; export type View = { text: string }; export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };',
    'reduce.nl': '---\nargs:\n  state: Counter\n  event: UiEvent\n  files: Record<string, File>\nreturns: Counter\n---\nRead files/note.txt/text and retain it while incrementing the count.',
    'view.ts': 'export default function view(state: Counter): View {\n  return { text: `${state.count}:${state.note}` };\n}',
    'note.txt': 'first note',
  };
  let fileInputs = 0;
  const client = new api.BrowserNatlangClient();
  const app = new api.BrowserNatlangApplication({ client,
    source: { files: project, reducer: 'reduce.nl', view: 'view.ts' },
    initialState: { count: 0, note: '' }, reducerInputs: () => {
      fileInputs++; return { files: api.textFileTree(project) };
    },
    modelTurn: request => {
      if (String(request.messages.at(-1)?.content ?? '').includes('return: complete'))
        return { calls: [], text: 'done', completion_tokens: 1 };
      const prior = request.messages.filter(message => message.role === 'assistant')
        .flatMap(message => message.tool_calls ?? []).at(-1)?.function.name;
      if (prior === 'eval') return { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
      return { calls: [['eval', { code: '({ count: state.count + 1, note: files["note.txt"].text })' }]], completion_tokens: 1 };
    } });
  try {
    assert.equal((await app.start()).view.text, '0:');
    assert.equal((await app.dispatch({ id: 'one', kind: 'request' })).view.text, '1:first note');
    project['note.txt'] = 'second note';
    assert.equal((await app.dispatch({ id: 'two', kind: 'request' })).view.text, '2:second note');
    assert.equal(fileInputs, 2);
  } finally { await app.close(); await client.close(); }
});

test('applications repair validation locally by default and may opt into caller feedback', async () => {
  const { BrowserNatlangApplication } = await api();
  const requests = [];
  const client = { run: async request => {
    requests.push(request);
    return { value: request.source.root === 'view.ts' ? { tag: 'p', text: 'ready' } : { count: 1 },
      outcome: { kind: 'done' }, trace: [], run_id: request.source.root };
  } };
  const app = new BrowserNatlangApplication({ client, source: { files, reducer: 'reduce.ts', view: 'view.ts' },
    initialState: { count: 0 } });
  try {
    await app.start();
    await app.dispatch({ id: 'one', kind: 'increment' });
    assert.deepEqual(requests.map(request => request.validationFeedback), ['local', 'local', 'local']);
  } finally { await app.close(); }
  requests.length = 0;
  const strict = new BrowserNatlangApplication({ client, source: { files, reducer: 'reduce.ts', view: 'view.ts' },
    initialState: { count: 0 }, validationFeedback: 'caller' });
  try {
    await strict.start();
    assert.deepEqual(requests.map(request => request.validationFeedback), ['caller']);
  } finally { await strict.close(); }
});

test('DOM renderer uses text nodes, emits typed events and rejects executable markup', async () => {
  const { BrowserDomRenderer } = await api();
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.listeners = new Map(); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    setAttribute(name, value) { this[name] = value; }
    emit(name) { this.listeners.get(name)?.(); }
  }
  const prior = globalThis.document;
  globalThis.document = { createElement: tag => new Element(tag) };
  const root = new Element('root'), events = [];
  const renderer = new BrowserDomRenderer(root, event => { events.push(event); });
  try {
    renderer.render({ tag: 'main', children: [
      { tag: 'p', text: '<script>alert(1)</script>' },
      { tag: 'input', id: 'entry', label: 'Entry', value: '' },
      { tag: 'button', text: 'Add', action: { kind: 'command', from: 'entry' } },
    ] });
    const [paragraph, input, button] = root.children[0].children;
    assert.equal(paragraph.textContent, '<script>alert(1)</script>');
    assert.equal(paragraph.children.length, 0);
    input.value = 'Buy milk'; button.emit('click');
    assert.deepEqual(events.map(event => [event.kind, event.value]),
      [['command', 'Buy milk']]);
    assert.equal(input['aria-label'], 'Entry');
    assert.throws(() => renderer.render({ tag: 'script', text: 'alert(1)' }), /unsupported/);
    assert.throws(() => renderer.render({ tag: 'button', action: {
      kind: 'command', from: 'missing' } }), /unknown input/);
    let deep = { tag: 'p', text: 'Reached the leaf' };
    for (let i = 0; i < 80; i++) deep = { tag: 'section', children: [deep] };
    renderer.render(deep);
    let leaf = root.children[0];
    for (let i = 0; i < 80; i++) leaf = leaf.children[0];
    assert.equal(leaf.textContent, 'Reached the leaf');
  } finally { renderer.close(); globalThis.document = prior; }
});

test('durable commit precedes view generation and refresh never replays a reducer', async () => {
  const { BrowserNatlangApplication } = await api();
  let reductions=0, views=0, failedView=false;const commits=[];
  const client={run:async request=>{
    if(request.source.root==='reduce.ts'){reductions++;return {value:{count:request.inputs.state.count+1},outcome:{kind:'done'},trace:[],run_id:'reduce'};}
    views++;if(failedView)throw new Error('presentation unavailable');
    return {value:{tag:'p',text:String(request.inputs.state.count)},outcome:{kind:'done'},trace:[],run_id:'view'};
  }};
  const app=new BrowserNatlangApplication({client,source:{files,reducer:'reduce.ts',view:'view.ts'},initialState:{count:4},initialRevision:7,onCommit:async record=>commits.push(record)});
  await app.start();failedView=true;
  await assert.rejects(app.dispatch({id:'durable',kind:'increment'}),/presentation unavailable/);
  assert.equal(commits[0].revision,8);assert.equal(commits[0].state.count,5);assert.equal(app.state.count,5);
  failedView=false;await app.refresh();assert.equal(reductions,1);assert.equal(views,3);assert.equal(app.view.text,'5');await app.close();
});

test('initial view failure keeps valid state available for refresh and later events',async()=>{
 const {BrowserNatlangApplication}=await api();let fail=true;
 const client={run:async request=>{
  if(request.source.root==='view.ts'&&fail)throw new Error('bad initial presentation');
  return {value:request.source.root==='view.ts'?{tag:'p',text:'ready'}:{count:request.inputs.state.count+1},outcome:{kind:'done'},trace:[],run_id:'test'};
 }};
 const app=new BrowserNatlangApplication({client,source:{files,reducer:'reduce.ts',view:'view.ts'},initialState:{count:0}});
 await assert.rejects(app.start(),/bad initial/);fail=false;await app.refresh();
 assert.equal((await app.dispatch({id:'after-failure',kind:'increment'})).state.count,1);await app.close();
});
