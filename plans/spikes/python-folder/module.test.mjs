import { loadPyodide } from 'pyodide';
import { readFileSync } from 'node:fs';
const py = await loadPyodide();
const seen = [];
// A stand-in host: nl records the type it was asked for; iterateOn runs a (Python) step until the predicate holds.
py.registerJsModule('_natlang_host', {
  nl: (instructions, returns) => { seen.push(returns); return async (...args) => returns === 'boolean' ? true : returns.startsWith('"') ? 'high' : `(${instructions})`; },
  iterate_on: (step, initial, ...fixed) => ({ until: async done => { let state = initial;
    for (let i = 0; i < 50 && !(await done(state)); i++) state = await step(state, ...fixed); return state; } }),
});
py.FS.writeFile('/lib/python3.14/site-packages/natlang.py', readFileSync(new URL('./natlang_py.py', import.meta.url), 'utf8'));
const out = await py.runPythonAsync(`
from typing import Literal, TypedDict
from natlang import nl, wait, iterate_on, type_text
class Ticket(TypedDict):
    id: str
    text: str
urgent = await nl[bool]("Is ticket urgent?")({"id": "T1", "text": "Checkout is down"})
level = await nl[Literal["low", "high"]]("Rate the ticket")("x")
labels = [wait(nl[bool]("Is it spam?")(m)) for m in ["a", "b"]]
final = await iterate_on(lambda n: n * 2, 1).until(lambda n: n > 100)
[urgent, level, labels, final, type_text(list[Ticket]), type_text(dict[str, int | None])]
`);
console.log(JSON.stringify(out.toJs()), '| types asked:', seen.join(' ; '));
