import type { EditorFile, EditorSnapshot, EditPatch, EditReport, CheckReport, RunReport, TraceView, ViewPanel, EditorView } from "../types.js";
import { host } from "natlang:runtime";

export default function inspect(): EditorSnapshot {
return host.ide.snapshot();
}
