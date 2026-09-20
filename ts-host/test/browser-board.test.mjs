import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}
const root = fileURLToPath(new URL('../examples/browser-board/program/', import.meta.url));
const names = ['board/reduce.nl', 'board/reduce/choose.nl', 'board/reduce/apply.ts',
  'board/view.nl', 'board/view/describe.nl', 'board/view/layout.ts', 'board/types.ts'];

test('native browser board reduces commands and renders a complete safe view tree', async () => {
  const { BrowserNatlangClient, BrowserNatlangApplication } = await api();
  const files = Object.fromEntries(await Promise.all(names.map(async name =>
    [name, await readFile(`${root}${name}`, 'utf8')])));
  const client = new BrowserNatlangClient();
  let event = null, app;
  const modelTurn = turn => {
    if (turn.messages.filter(message => message.role === 'assistant').length > 1)
      return { calls: [], text: 'done', completion_tokens: 1 };
    const prompt = String(turn.messages.find(message => message.role === 'user')?.content ?? '');
    if (prompt.includes('function reduce(')) return { calls: [
      ['call', { function: 'choose', to: 'let/decision', inputs: {
        state: 'args/state', event: 'args/event' } }],
      ['call', { function: 'apply', to: 'return', inputs: {
        state: 'args/state', event: 'args/event', decision: 'let/decision' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('function view(')) return { calls: [
      ['call', { function: 'describe', to: 'let/plan', inputs: { state: 'args/state' } }],
      ['call', { function: 'layout', to: 'return', inputs: {
        state: 'args/state', plan: 'let/plan' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('Interpret the incoming UI event')) {
      const decision = event.kind === 'command' ? { kind: 'add', text: event.value } :
        { kind: 'toggle', item_id: event.value };
      return { calls: [['write', { path: 'return', value: decision }]], completion_tokens: 1 };
    }
    const state = app?.state ?? { revision: 0, next_id: 1, items: [] };
    return { calls: [['write', { path: 'return', value: {
      title: 'Tasks', summary: `${state.items.length} tasks`, groups: [
        { label: 'Open', ids: state.items.filter(item => !item.done).map(item => item.id) },
        { label: 'Done', ids: state.items.filter(item => item.done).map(item => item.id) },
      ],
    } }]], completion_tokens: 1 };
  };
  app = new BrowserNatlangApplication({ client,
    source: { files, reducer: 'board/reduce.nl', view: 'board/view.nl' },
    initialState: { revision: 0, next_id: 1, items: [] }, modelTurn });
  try {
    const first = await app.start();
    assert.equal(first.view.tag, 'main');
    event = { id: 'one', kind: 'command', value: '<img src=x onerror=alert(1)>' };
    const added = await app.dispatch(event);
    assert.equal(added.state.items[0].text, '<img src=x onerror=alert(1)>');
    const list = added.view.children.find(node => node.tag === 'section');
    assert.equal(list.children[1].children[0].children[0].text,
      '<img src=x onerror=alert(1)>');
    event = { id: 'two', kind: 'toggle', value: 'task-1' };
    const toggled = await app.dispatch(event);
    assert.equal(toggled.state.items[0].done, true);
    assert.equal(toggled.view.children.at(-1).children[1].children[0].children[1].action.value,
      'task-1');
  } finally { await app.close(); await client.close(); }
});
