import { BrowserLocalModel, BrowserNatlangHost, BrowserNatlangClient,
  BrowserNatlangApplication, BrowserDomRenderer,
  NativeSourceWorkspace, loadFunctionFiles,
  type BrowserRunRequest, type UiNode } from '../dist/browser/index.js';

const request: BrowserRunRequest = {
  source: { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>', code: 'return 1;' } } },
};
const host = new BrowserNatlangHost();
void host.run(request);
host.close();

const model = new BrowserLocalModel();
const localHost = new BrowserNatlangHost({ model });
void model.loadFromHuggingFace({ repo: 'Qwen/Qwen3-0.6B-GGUF', quant: 'Q4_K_M' });
void localHost.run(request);
localHost.close();
void model.close();

const client = new BrowserNatlangClient({ host: { count: 1 }, mode: 'retained' });
void client.loadModel({ kind: 'url', url: '/models/natlang.gguf',
  templateUrl: '/models/template.jinja' }, { contextTokens: 4096 });
void client.run(request);
void client.close();

const view: UiNode = { tag: 'button', text: 'Go', action: { kind: 'go' } };
const application = new BrowserNatlangApplication({ client,
  source: { files: { 'reduce.ts': '', 'view.ts': '' }, reducer: 'reduce.ts', view: 'view.ts' },
  initialState: { count: 0 }, onTransition: transition => { void transition.view; } });
void application.dispatch({ id: 'one', kind: 'go' });
void application.close();
void BrowserDomRenderer;
void view;

const files = { 'main.nl': '---\nreturns: Num\n---\nReturn one.' };
void loadFunctionFiles('main.nl', files);
void localHost.run({ source: { kind: 'files', root: 'main.nl', files } });
const workspace = new NativeSourceWorkspace({ leaf: { returns: 'Num', instructions: 'Return one.' } }, 'leaf');
void workspace.invoke('leaf');
