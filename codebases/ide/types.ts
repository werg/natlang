export type EditorFile = { name: string, kind: string, source: string, returns: string };
export type EditorSnapshot = { revision: string, root: string, files: EditorFile[] };
export type EditPatch = { name: string, start: number, end: number, text: string,
  expected_revision: string };
export type EditReport = { status: string, revision: string, detail: string };
export type CheckReport = { status: string, revision: string, source_revision: string, detail: string };
export type RunReport = { status: string, run_id: string, revision: string,
  source_revision: string, value_text: string, trace_events: number, detail: string };
export type TraceView = { run_id: string, source_revision: string,
  index: number, total: number, event_json: string };
export type ViewPanel = { heading: string, body: string };
export type EditorView = { title: string, panels: ViewPanel[] };
