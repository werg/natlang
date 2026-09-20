import { BrowserNatlangHost, type BrowserRunRequest } from '../dist/browser/index.js';

const request: BrowserRunRequest = {
  source: { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>', code: 'return 1;' } } },
};
const host = new BrowserNatlangHost();
void host.run(request);
host.close();
