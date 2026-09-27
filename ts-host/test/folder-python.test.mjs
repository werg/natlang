import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, runFolderPython } from '../dist/index.js';

test('Python reads and writes the folder through pathlib and sqlite', async () => {
  const folder = Folder.fromFiles({ 'notes/a.txt': 'alpha' });
  const result = await runFolderPython(folder, `
from pathlib import Path
import sqlite3
Path('out').mkdir()
Path('out/b.txt').write_text(Path('notes/a.txt').read_text().upper())
db = sqlite3.connect('out/items.db')
db.executescript('create table t(n integer); insert into t values (4);')
answer = db.execute('select n from t').fetchone()[0]
db.close()
answer
`);
  assert.equal(result.value, 4);
  assert.equal(await folder.readText('out/b.txt'), 'ALPHA');
  assert.deepEqual(result.changedPaths.sort(), ['out/b.txt', 'out/items.db']);
});

test('Python policy rejects open loops and allows finite comprehensions', async () => {
  const folder = Folder.fromFiles({});
  await assert.rejects(runFolderPython(folder, 'while True: pass'), /while.*not allowed/);
  assert.equal((await runFolderPython(folder, 'sum(i for i in range(4))')).value, 6);
});

test('Python natlang facade calls the supplied child hook', async () => {
  const folder = Folder.fromFiles({});
  const types = [];
  const result = await runFolderPython(folder, `
from natlang import nl
await nl[bool]('Is the message urgent?')('please help now')
`, { nl: (_instructions, returns) => async message => { types.push(returns); return message.includes('help'); } });
  assert.equal(result.value, true);
  assert.deepEqual(types, ['boolean']);
});

test('vendored pandas loads and reads a CSV from the folder', async () => {
  const folder = Folder.fromFiles({ 'sales.csv': 'amount\n4\n5\n' });
  const result = await runFolderPython(folder, 'import pandas as pd\npd.read_csv("sales.csv").amount.sum()');
  assert.equal(result.value, 9);
});

test('the watchdog interrupts a long pure-Python computation', async () => {
  const folder = Folder.fromFiles({});
  await assert.rejects(runFolderPython(folder, 'for i in range(10**10): pass', {}, 100), /timed out/);
});

test('Python refuses a cell that leaves a writable file open', async () => {
  const folder = Folder.fromFiles({});
  await assert.rejects(runFolderPython(folder, 'f = open("unfinished.txt", "w"); f.write("draft")'),
    /close the Python file or sqlite connection/);
});
