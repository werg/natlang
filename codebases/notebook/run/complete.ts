import type { Cell, File, CellResult, NotebookState } from "../types.js";

export default function complete(state: NotebookState): boolean {
return state.status !== 'running';
}
