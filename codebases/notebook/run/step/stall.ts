import type { Cell, File, CellResult, NotebookState } from "../../types.js";

export default function stall(state: NotebookState): NotebookState {
const done = new Set(state.order);
const blocked = state.cells.filter(cell => !done.has(cell.id)).map(cell => cell.id).sort();
return { ...state, status: 'blocked', blocked,
  detail: 'No required cell is ready; check missing dependencies or a cycle.' };
}
