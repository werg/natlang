/*---
engine: typescript-host
args:
  goal: string
returns: NotebookState
---*/
try {
  const cells = host.notebook.describe(args.goal);
  const ids = new Set(cells.map(cell => cell.id));
  if (ids.size !== cells.length || cells.some(cell => cell.needs.includes(cell.id) ||
      new Set(cell.needs).size !== cell.needs.length))
    throw new Error('duplicate cell or self dependency');
  return { goal: args.goal, cells, order: [], results: [], blocked: [],
    status: 'running', detail: '', answer: '' };
} catch (error) {
  return { goal: args.goal, cells: [], order: [], results: [], blocked: [],
    status: 'invalid', detail: String(error), answer: '' };
}
