import { loadPyodide } from 'pyodide';
import { readFileSync } from 'node:fs';
const py = await loadPyodide();
py.FS.mkdirTree('/work'); py.FS.writeFile('/work/helpers.py', 'def total(xs):\n    s = 0\n    while xs: s += xs.pop()\n    return s\n');
py.FS.writeFile('/work/ok_helpers.py', 'def total(xs):\n    return sum(x for x in xs)\n');
py.FS.writeFile('/lib/python3.14/site-packages/natlang_policy.py', readFileSync(new URL('./natlang_policy.py', import.meta.url), 'utf8'));
py.runPython(`import natlang_policy, os; os.chdir('/work'); natlang_policy.install('/work'); ns = natlang_policy.fresh_namespace()`);
const cells = {
  while: 'while True: pass',
  recursion: 'def f(n):\n    return f(n - 1) if n else 0',
  mutual: 'def a(n): return b(n)\ndef b(n): return a(n)',
  exec: 'exec("1")',
  sneaky: 'f = __builtins__["ex" + "ec"]',
  dataclass: 'from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int\nresult = P(3).x',
  js: 'import js',
  jsfrom: 'from pyodide.ffi import to_js',
  endless: 'import itertools\nfor i in itertools.count(): pass',
  capped: 'import itertools\nfirst = [i for i in itertools.islice(itertools.count(), 5)]\nresult = first',
  generator: 'result = [x * 2 for x in (y for y in range(3))]',
  zip: 'result = [a + b for a, b in zip([1, 2], [3, 4])]',
  localbad: 'import helpers',
  localok: 'import ok_helpers\nresult = ok_helpers.total([1, 2, 3])',
  pandas_ok: 'result = [len(k) for k in {"ab": 1, "c": 2}]',
};
for (const [name, source] of Object.entries(cells)) {
  py.globals.set('source', source);
  try { py.runPython(`ns.pop('result', None); natlang_policy.run_cell(source, ns); ns.get('result')`);
    console.log(`${name.padEnd(10)} ok     ${JSON.stringify(py.runPython("ns.get('result')")?.toJs?.() ?? py.runPython("ns.get('result')"))}`); }
  catch (e) { console.log(`${name.padEnd(10)} REFUSED ${String(e.message).trim().split('\n').slice(-1)[0].slice(0, 150)}`); }
}
