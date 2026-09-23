import type { EditorFile, EditorSnapshot, EditPatch, EditReport, CheckReport, RunReport, TraceView, ViewPanel, EditorView } from "../types.js";
import { host } from "natlang:runtime";

export default function apply(patch: EditPatch): EditReport {
return host.ide.edit(patch);
}
