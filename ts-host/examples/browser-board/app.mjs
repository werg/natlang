import { BrowserNatlangClient, BrowserNatlangApplication, BrowserDomRenderer,
  loadBrowserModelCatalog } from '../../dist/browser/natlang.js';

const $ = id => document.getElementById(id);
const files = {};
const names = [
  'board/reduce.nl', 'board/reduce/choose.nl', 'board/reduce/apply.ts',
  'board/view.nl', 'board/view/describe.nl', 'board/view/layout.ts', 'board/types.ts',
];
const client = new BrowserNatlangClient({
  wasmUrl: '/ts-host/dist/browser/wllama.wasm',
  compatWorkerUrl: '/ts-host/dist/browser/wllama-compat.js',
  compatWasmUrl: '/ts-host/dist/browser/wllama-compat.wasm',
});
let app, renderer, currentEvent = null;
let uiQueue = Promise.resolve();
const status = text => { $('status').textContent = text; };
const initial = () => ({ revision: 0, next_id: 1, items: [] });
const catalog = await loadBrowserModelCatalog();
for (const model of catalog.models)
  $('model').add(new Option(model.label, model.id));
$('model').value = catalog.defaultId;

async function source() {
  if (!Object.keys(files).length) for (const name of names) {
    const response = await fetch(`./program/${name}`);
    if (!response.ok) throw new Error(`missing application source: ${name}`);
    files[name] = await response.text();
  }
  return { files, reducer: 'board/reduce.nl', view: 'board/view.nl' };
}

function fixtureTurn(turn) {
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
    const event = currentEvent;
    const decision = event?.kind === 'command' ? { kind: 'add', text: event.value } :
      event?.kind === 'toggle' ? { kind: 'toggle', item_id: event.value } :
        event?.kind === 'clear_done' ? { kind: 'clear_done' } : { kind: 'ignore' };
    return { calls: [['write', { path: 'return', value: decision }]], completion_tokens: 1 };
  }
  const state = app?.state ?? initial();
  return { calls: [['write', { path: 'return', value: {
    title: 'Today’s board', summary: `${state.items.length} tasks`,
    groups: [{ label: 'Open', ids: state.items.filter(item => !item.done).map(item => item.id) },
      { label: 'Done', ids: state.items.filter(item => item.done).map(item => item.id) }],
  } }]], completion_tokens: 1 };
}

async function start(fixture = false) {
  renderer?.close();
  if (app) await app.close();
  app = new BrowserNatlangApplication({ client, source: await source(), initialState: initial(),
    seedRoot: 17, ...(fixture ? { modelTurn: fixtureTurn } : {}),
    onTransition: transition => {
      renderer.render(transition.view);
      $('detail').textContent = JSON.stringify({ revision: transition.revision,
        event: transition.event, reducerRun: transition.reducerRun?.run_id,
        viewRun: transition.viewRun.run_id,
        reducerTrace: transition.reducerRun?.trace.length,
        viewTrace: transition.viewRun.trace.length }, null, 2);
      status(`Revision ${transition.revision} ready`);
    },
    onFailure: failure => status(`${failure.stage} failed: ${failure.detail}`),
  });
  renderer = new BrowserDomRenderer($('app'), event => {
    const next = uiQueue.then(async () => {
      currentEvent = event;
      status(`Reducing ${event.kind}…`);
      await app.dispatch(event);
    });
    uiQueue = next.catch(() => undefined);
    return next;
  }, error => status(String(error)));
  await app.start();
  window.natlangBoard = { app, client };
}

$('load').onclick = async () => {
  $('load').disabled = true;
  try {
    const selected = catalog.models.find(row => row.id === $('model').value);
    if (!selected) throw new Error('choose a model');
    status('Loading browser model…');
    await client.loadModel({ kind: 'url', id: selected.id, url: selected.url,
      templateUrl: selected.templateUrl }, { contextTokens: 8192 });
    await start();
  } catch (error) { status(String(error)); }
  finally { $('load').disabled = false; }
};

if (new URLSearchParams(location.search).has('fixture'))
  start(true).catch(error => status(String(error)));
