import type { EditorFile, EditorSnapshot, EditPatch, EditReport, CheckReport, RunReport, TraceView, ViewPanel, EditorView } from "../types.js";
import { host } from "natlang:runtime";

export default function trace(run_id: string, index: number): TraceView {
return host.ide.inspect(run_id, index);
}
