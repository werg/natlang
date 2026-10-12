import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { WATCHDOG_SCRIPT } from '../dist/native/folder-python.js';
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

test('the watchdog also interrupts C code draining an endless iterator', async () => {
  const folder = Folder.fromFiles({});
  for (const code of ['import itertools\nsum(itertools.count())', 'from itertools import repeat\nany(repeat(0))',
    'import collections, itertools\ncollections.deque(itertools.cycle([1]), maxlen=0)', 'x = []\nsum(iter(x.__len__, 1))'])
    await assert.rejects(runFolderPython(folder, code, {}, 100), /timed out/, code);
  const same = 'import itertools\n[list(itertools.islice(itertools.count(5, 2), 3)), list(itertools.islice(itertools.cycle("ab"), 3)), ' +
    'list(zip("ab", itertools.repeat(0))), list(iter(iter([1, 2, 0, 3]).__next__, 0))]';
  assert.deepEqual((await runFolderPython(folder, same)).value, [[5, 7, 9], ['a', 'b', 'a'], [['a', 0], ['b', 0]], [1, 2]]);
});

test('Python refuses a cell that leaves a writable file open', async () => {
  const folder = Folder.fromFiles({});
  await assert.rejects(runFolderPython(folder, 'f = open("unfinished.txt", "w"); f.write("draft")'),
    /close the Python file or sqlite connection/);
});


test('the watchdog reasserts a consumed interrupt until termination', async () => {
  const buffer = new SharedArrayBuffer(4);
  const flag = new Int32Array(buffer);
  const worker = new Worker(`const { workerData } = require('node:worker_threads');
    const { buffer, ms } = workerData; ${WATCHDOG_SCRIPT}`, { eval: true, workerData: { buffer, ms: 10 } });
  try {
    for (let attempt = 0; attempt < 3; attempt++) {
      Atomics.store(flag, 0, 0);
      // Poll from another event loop, just as Pyodide consumes its signal slot.
      const deadline = Date.now() + 1000;
      while (Atomics.load(flag, 0) !== 2 && Date.now() < deadline)
        await new Promise(resolve => setTimeout(resolve, 5));
      assert.equal(Atomics.load(flag, 0), 2, `interrupt request ${attempt}`);
    }
  } finally { await worker.terminate(); }
});
