/*---
engine: typescript-host
args:
  state: NotebookState
returns: Cell[]
---*/
const byId = new Map(args.state.cells.map(cell => [cell.id, cell]));
const needed = new Set();
const visit = id => { if (needed.has(id)) return; needed.add(id);
  for (const parent of byId.get(id)?.needs ?? []) visit(parent); };
visit(args.state.goal);
const done = new Set(args.state.order);
return args.state.cells.filter(cell => needed.has(cell.id) && !done.has(cell.id) &&
  cell.needs.every(parent => done.has(parent)));
