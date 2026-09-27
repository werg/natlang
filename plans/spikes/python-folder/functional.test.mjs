import { loadPyodide } from 'pyodide';
import { Folder } from '../../../ts-host/dist/native/scoped-fs.js';
import { folderFS, folderView } from './folderfs.mjs';
const folder = Folder.fromFiles({ 'notes/a.md': 'status: open\n', 'notes/b.md': 'status: done\n', 'sales.csv': 'region,amount\nnorth,10\nsouth,5\nnorth,7\n' });
const py = await loadPyodide();
await py.loadPackage(['pandas'], { messageCallback: () => {} });
try { py.FS.mkdir("/work"); py.FS.mount(folderFS(py, folderView(folder)), {}, "/work"); } catch (e) { console.log("MOUNT", e?.message ?? e, String(e?.stack ?? "").split("\n").slice(0,4).join(" | ").slice(0, 600)); process.exit(1); }
const out = await py.runPythonAsync(`
import os, pathlib, shutil, sqlite3, pandas as pd
os.chdir('/work'); log = []
log.append(sorted(os.listdir('.')))
log.append(open('notes/a.md').read())
with open('notes/a.md', 'a') as f: f.write('owner: Ada\\n')
log.append(sorted(str(p) for p in pathlib.Path('.').rglob('*.md')))
log.append([ (r, sorted(d), sorted(f)) for r, d, f in os.walk('.') ])
os.makedirs('out/deep', exist_ok=True)
df = pd.read_csv('sales.csv'); df.groupby('region').amount.sum().to_csv('out/deep/totals.csv')
os.rename('notes/b.md', 'notes/done-b.md'); os.remove('notes/done-b.md')
shutil.copy('sales.csv', 'out/sales-copy.csv')
con = sqlite3.connect('db.sqlite'); con.execute('create table t(a)'); con.executemany('insert into t values (?)', [(i,) for i in range(100)]); con.commit()
log.append(con.execute('select sum(a) from t').fetchone()[0]); con.close()
try: open('missing.txt')
except FileNotFoundError as e: log.append('FileNotFoundError')
log.append(os.stat('sales.csv').st_size)
log
`);
console.log(JSON.stringify(out.toJs(), null, 0));
console.log('folder now:', folder.listFiles('').map(f => `${f.path}:${f.size}`).join(' '));
console.log('a.md =', JSON.stringify(await folder.readText('notes/a.md')));
console.log('totals =', JSON.stringify(await folder.readText('out/deep/totals.csv')));
console.log('diff:', JSON.stringify((await folder.diff('')).map?.(c => `${c.kind}:${c.path}`) ?? await folder.diff('')));
