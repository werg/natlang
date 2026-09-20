import { BrowserLocalModel, BrowserNatlangHost, NativeSourceWorkspace, loadFunctionFiles,
  type BrowserRunRequest } from '../dist/browser/index.js';

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

const files = { 'main.nl': '---\nreturns: Num\n---\nReturn one.' };
void loadFunctionFiles('main.nl', files);
void localHost.run({ source: { kind: 'files', root: 'main.nl', files } });
const workspace = new NativeSourceWorkspace({ leaf: { returns: 'Num', instructions: 'Return one.' } }, 'leaf');
void workspace.invoke('leaf');
