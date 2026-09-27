import { loadPyodide } from 'pyodide';
import { Worker } from 'node:worker_threads';
import { Folder } from '../../../ts-host/dist/native/scoped-fs.js';
import { folderFS, folderView } from './folderfs.mjs';
// A source that counts content reads, over 10,000 files.
let reads = 0;
const paths = Array.from({ length: 10000 }, (_, i) => `mail/${i % 50}/${i}.eml`);
const known = new Set(paths);
const source = { size: path => `Subject: message ${path}\n\nbody\n`.length, paths: () => paths, get: path => { if (!known.has(path)) return undefined; reads++; return new TextEncoder().encode(`Subject: message ${path}\n\nbody\n`); } };
const folder = new Folder(source, 'write'), view = folderView(folder);
const py = await loadPyodide();
await py.loadPackage(['pandas'], { messageCallback: () => {} });
py.FS.mkdir('/work');
let fsType;
const mount = () => { fsType = folderFS(py, view); py.FS.mount(fsType, {}, '/work'); };
mount();
py.registerJsModule('natlang', { nl: instructions => async value => { await new Promise(r => setTimeout(r, 5)); return String(value).length % 2 === 0; } });
const time = async (label, code) => { const t = Date.now(); reads = 0; const r = await py.runPythonAsync(code);
  console.log(`${label}: ${Date.now() - t} ms, content reads ${reads}, ->`, JSON.stringify(r?.toJs ? r.toJs() : r)?.slice(0, 120)); };
await time('listdir(mail)', `import os; os.chdir('/work'); len(os.listdir('mail'))`);
await time('walk 10k', `sum(len(f) for _, _, f in os.walk('mail'))`);
await time('stat 10k', `sum(os.stat(os.path.join(r, n)).st_size for r, _, fs in os.walk('mail') for n in fs)`);
await time('read 1k', `sum(len(open(f'mail/{i % 50}/{i}.eml').read()) for i in range(1000))`);
// Coherence: the folder changes outside Python; a fresh mount shows it.
await folder.writeText('mail/0/0.eml', 'changed outside\n'); view.external(); fsType.refresh();   // what entering Python does
await time('changed outside, same mount', `open('mail/0/0.eml').read()`);
// Synchronous nl from inside pandas.
await time('nl in pandas apply', `
import pandas as pd
from pyodide.ffi import run_sync
from natlang import nl
df = pd.DataFrame({'t': ['a', 'bb', 'ccc', 'dddd']})
df['even'] = df.t.apply(lambda t: run_sync(nl('Is it even?')(t)))
df.even.tolist()`);
// Interrupt a long pure-Python loop from a watchdog thread.
const buffer = new SharedArrayBuffer(4); py.setInterruptBuffer(new Int32Array(buffer));
new Worker(new URL('./watchdog.mjs', import.meta.url), { workerData: { buffer, ms: 500 } });
await time('interrupt', `
def spin():
    try:
        total = 0
        for i in range(10**10): total += i
        return 'not interrupted'
    except KeyboardInterrupt:
        return 'interrupted'
spin()`);
process.exit(0);
