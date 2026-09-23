import { BrowserDomRenderer, BrowserLocalModel, EventLoop, compileVirtualProject, createNatlangRuntime, iterateOn,
  loadBrowserLocalModel, loadVirtualNatlang, nl, type UiNode } from '../dist/browser/index.js';
import * as natlang from '../dist/browser/index.js';

const model = new BrowserLocalModel();
void model.loadFromHuggingFace({ repo: 'Qwen/Qwen3-0.6B-GGUF', quant: 'Q4_K_M' });
void loadBrowserLocalModel({ kind: 'url', url: '/models/natlang.gguf', templateUrl: '/models/template.jinja' }, { contextTokens: 4096 })
  .then(loaded => loaded.status.id);
const runtime = createNatlangRuntime({ model: model.turn, services: { counter: { count: 1 } } });

const add = loadVirtualNatlang({ 'add.nl': '---\nargs:\n  a: number\nreturns: number\n---\nAdd one to a.\n' }, 'add.nl');
void runtime.run(() => add(1));
const project = compileVirtualProject({ files: { 'main.ts': 'export const one = 1;\n' } }, natlang);
void project.require('main.ts');

const view: UiNode = { tag: 'button', text: 'Go', action: { kind: 'go' } };
const loop = new EventLoop<{ count: number }, UiNode>({ initialState: { count: 0 }, reduce: state => ({ count: state.count + 1 }),
  view: () => view, step: fn => runtime.run(fn) });
void loop.dispatch({ id: 'one', kind: 'go' });
void loop.close();
void BrowserDomRenderer;
void nl;
void iterateOn(async (value: number) => value + 1, 0).until(value => value > 2);
