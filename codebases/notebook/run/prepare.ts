import type { Cell, File, CellResult, NotebookState } from "../types.js";
import { host } from "natlang:runtime";

export default function prepare(goal: string): NotebookState {
try {
  const cells = host.notebook.describe(goal);
  const ids = new Set(cells.map(cell => cell.id));
  if (ids.size !== cells.length || cells.some(cell => cell.needs.includes(cell.id) ||
      new Set(cell.needs).size !== cell.needs.length))
    throw new Error('duplicate cell or self dependency');
  return { goal: goal, cells, order: [], results: [], blocked: [],
    status: 'running', detail: '', answer: '' };
} catch (error) {
  return { goal: goal, cells: [], order: [], results: [], blocked: [],
    status: 'invalid', detail: String(error), answer: '' };
}
}
