import type { EditorFile, EditorSnapshot, EditPatch, EditReport, CheckReport, RunReport, TraceView, ViewPanel, EditorView } from "../types.js";

export default function invalid(revision: string, checked: CheckReport): RunReport {
return { status: 'invalid-source', run_id: '', revision: revision,
  source_revision: '',
  value_text: '', trace_events: 0, detail: checked.detail };
}
