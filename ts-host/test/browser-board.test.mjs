import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { scriptedModel } from './support/natlang.mjs';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('the browser board compiles in the page, reduces commands, and renders a complete safe view tree', async () => {
  const natlang = await api();
  const source = await readFile(new URL('../examples/browser-board/program/board.ts', import.meta.url), 'utf8');
  const compiled = natlang.compileVirtualProject({ files: { 'board.ts': source } }, natlang);
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  const board = compiled.require('board.ts');
  const model = scriptedModel(opening => opening.includes('Interpret the UI event') ?
    'return event.kind === "command" ? { kind: "add", text: event.value } : { kind: "toggle", item_id: event.value }' :
    'return { title: "Tasks", summary: `${state.items.length} tasks`, groups: [' +
    '{ label: "Open", ids: state.items.filter(item => !item.done).map(item => item.id) }, ' +
    '{ label: "Done", ids: state.items.filter(item => item.done).map(item => item.id) }] }');
  const runtime = natlang.createNatlangRuntime({ model: model.driver });
  const loop = new natlang.EventLoop({ initialState: board.initialBoard(), reduce: board.reduce, view: board.view,
    step: fn => runtime.run(fn) });
  try {
    const first = await loop.start();
    assert.equal(first.view.tag, 'main');
    const added = await loop.dispatch({ id: 'one', kind: 'command', value: '<img src=x onerror=alert(1)>' });
    assert.equal(added.state.items[0].text, '<img src=x onerror=alert(1)>');
    const list = added.view.children.find(node => node.tag === 'section');
    assert.equal(list.children[1].children[0].children[0].text, '<img src=x onerror=alert(1)>');
    const toggled = await loop.dispatch({ id: 'two', kind: 'toggle', value: 'task-1' });
    assert.equal(toggled.state.items[0].done, true);
    assert.equal(toggled.view.children.at(-1).children[1].children[0].children[1].action.value, 'task-1');
    assert.match(model.openings[0], /state: Board/);
  } finally { await loop.close(); }
});
