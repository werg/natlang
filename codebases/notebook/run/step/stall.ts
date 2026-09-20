/*---
engine: typescript-host
args:
  state: NotebookState
returns: NotebookState
---*/
const done = new Set(args.state.order);
const blocked = args.state.cells.filter(cell => !done.has(cell.id)).map(cell => cell.id).sort();
return { ...args.state, status: 'blocked', blocked,
  detail: 'No required cell is ready; check missing dependencies or a cycle.' };
