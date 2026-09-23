import type { Cell, File, CellResult, NotebookState } from "../../types.js";

export default function ready_cells(state: NotebookState): Cell[] {
const byId = new Map(state.cells.map(cell => [cell.id, cell]));
const needed = new Set();
const visit = id => { if (needed.has(id)) return; needed.add(id);
  for (const parent of byId.get(id)?.needs ?? []) visit(parent); };
visit(state.goal);
const done = new Set(state.order);
return state.cells.filter(cell => needed.has(cell.id) && !done.has(cell.id) &&
  cell.needs.every(parent => done.has(parent)));
}
