/*---
engine: typescript-host
args:
  state: NotebookState
  chosen: string
returns: NotebookState
---*/
const state = args.state, done = new Set(state.order);
const cell = state.cells.find(row => row.id === args.chosen);
if (!cell || done.has(cell.id) || !cell.needs.every(parent => done.has(parent)))
  return { ...state, status: 'invalid', detail: `cell is not ready: ${args.chosen}` };
const result = await host.notebook.execute(cell.id);
const results = [...state.results, result];
if (result.revision !== cell.revision)
  return { ...state, results, status: 'stale', detail: `${cell.id}: source revision changed during this run` };
if (result.status !== 'ok') return { ...state, results,
  status: result.status, detail: `${cell.id}: ${result.detail}` };
const order = [...state.order, cell.id];
return { ...state, order, results,
  status: cell.id === state.goal ? 'done' : 'running', detail: '' };
