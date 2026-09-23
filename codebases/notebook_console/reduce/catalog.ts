import type { Cell, CellResult, NotebookState, ConsoleEvent, File, ConsoleState, ViewBlock, TerminalView } from "../types.js";
import { host } from "natlang:runtime";

export default function catalog(): Cell[] {
return host.notebook.catalog();
}
