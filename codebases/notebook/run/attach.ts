import type { Cell, File, CellResult, NotebookState } from "../types.js";

export default function attach(state: NotebookState, answer: string): NotebookState {
return { ...state, answer: answer };
}
