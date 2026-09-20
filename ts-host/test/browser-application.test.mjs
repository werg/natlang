import assert from 'node:assert/strict';
import { test } from 'node:test';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

const files = {
  'types.ts': 'export type Counter = { count: Num };\nexport type UiEvent = { id: Text, kind: Text, value?: Text };\nexport type View = { tag: Text, text: Text };',
  'reduce.ts': '/*---\nengine: typescript-host\nargs:\n  state: Counter\n  event: UiEvent\nreturns: Counter\n---*/\nif (args.event.kind === "bad") return { count: "wrong" };\nreturn { count: args.state.count + 1 };',
  'view.ts': '/*---\nengine: typescript-host\nargs:\n  state: Counter\nreturns: View\n---*/\nreturn { tag: "p", text: `Count ${args.state.count}` };',
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
  let rendered = 0;
  const app = new BrowserNatlangApplication({ client,
    source: { files: naturalFiles, reducer: 'reduce.ts', view: 'view.nl' },
    initialState: { count: 0 }, modelTurn: turn => {
      if (turn.messages.filter(message => message.role === 'assistant').length > 1)
        return { calls: [], text: 'done', completion_tokens: 1 };
      return { calls: [['write', { path: 'return', value: {
        tag: 'p', text: `Natlang view ${rendered++}` } }]], completion_tokens: 1 };
    } });
  try {
    assert.equal((await app.start()).view.text, 'Natlang view 0');
    assert.equal((await app.dispatch({ id: 'one', kind: 'increment' })).view.text,
      'Natlang view 1');
    assert.equal(app.state.count, 1);
  } finally { await app.close(); await client.close(); }
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
  } finally { renderer.close(); globalThis.document = prior; }
});
