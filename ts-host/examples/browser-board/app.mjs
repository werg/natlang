import * as natlang from '../../dist/browser/natlang.js';
const { BrowserDomRenderer, EventLoop, compileVirtualProject, createNatlangRuntime, formatDiagnostics,
  loadBrowserLocalModel, loadBrowserModelCatalog } = natlang;

const $ = id => document.getElementById(id);
const status = text => { $('status').textContent = text; };
const catalog = await loadBrowserModelCatalog();
for (const model of catalog.models) $('model').add(new Option(model.label, model.id));
$('model').value = catalog.defaultId;

/** The board program, compiled in the page with the same compiler as a Node build. */
async function program() {
  const response = await fetch('./program/board.ts');
  if (!response.ok) throw new Error('missing application source: board.ts');
  const compiled = compileVirtualProject({ files: { 'board.ts': await response.text() } }, natlang);
  if (!compiled.ok) throw new Error(formatDiagnostics(compiled.diagnostics));
  return compiled.require('board.ts');
}

/** Scripted decisions for wiring checks (`?fixture`); they do not assess a model. */
function fixtureModel(getState, getEvent) {
  return ({ messages }) => {
    if (messages.length > 2) return { text: 'done' };
    const opening = String(messages[1].content);
    const event = getEvent(), state = getState();
    const value = opening.includes('Interpret the UI event') ?
      (event?.kind === 'command' ? { kind: 'add', text: event.value } : event?.kind === 'toggle' ? { kind: 'toggle', item_id: event.value } :
        event?.kind === 'clear_done' ? { kind: 'clear_done' } : { kind: 'ignore' }) :
      { title: 'Today’s board', summary: `${state.items.length} tasks`, groups: [
        { label: 'Open', ids: state.items.filter(item => !item.done).map(item => item.id) },
        { label: 'Done', ids: state.items.filter(item => item.done).map(item => item.id) }] };
    const lines = [...opening.matchAll(/^\s*(\d+) \[ \]/gm)].map(match => Number(match[1]));
    return { calls: [['eval', { code: `result = ${JSON.stringify(value)}` }], ['mark_lines', { start: 1, end: Math.max(1, ...lines) }]] };
  };
}

let loop, renderer, currentEvent = null, uiQueue = Promise.resolve();
async function start(model) {
  renderer?.close();
  await loop?.close();
  const board = await program();
  const runtime = createNatlangRuntime({ model: model ?? fixtureModel(() => loop?.state ?? board.initialBoard(), () => currentEvent) });
  loop = new EventLoop({ initialState: board.initialBoard(), reduce: board.reduce, view: board.view,
    step: (fn, context) => runtime.run(fn, { signal: context.signal, name: context.stage }),
    onTransition: transition => {
      renderer.render(transition.view);
      $('detail').textContent = JSON.stringify({ revision: transition.revision, event: transition.event }, null, 2);
      status(`Revision ${transition.revision} ready`);
    },
    onFailure: failure => status(`${failure.stage} failed: ${failure.error}`) });
  renderer = new BrowserDomRenderer($('app'), event => {
    const next = uiQueue.then(async () => { currentEvent = event; status(`Reducing ${event.kind}…`); await loop.dispatch(event); });
    uiQueue = next.catch(() => undefined);
    return next;
  }, error => status(String(error)));
  await loop.start();
  window.natlangBoard = { loop, runtime };
}

$('load').onclick = async () => {
  $('load').disabled = true;
  try {
    const selected = catalog.models.find(row => row.id === $('model').value);
    if (!selected) throw new Error('choose a model');
    status('Loading browser model…');
    const { model } = await loadBrowserLocalModel({ kind: 'url', id: selected.id, url: selected.url, templateUrl: selected.templateUrl },
      { contextTokens: 8192 });
    await start(model.turn);
  } catch (error) { status(String(error)); }
  finally { $('load').disabled = false; }
};

if (new URLSearchParams(location.search).has('fixture')) start().catch(error => status(String(error)));
