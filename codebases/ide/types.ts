export type EditorFile = { name: Text, kind: Text, source: Text, returns: Text };
export type File = { kind: Text, text?: Text, bytes: Num };
export type EditorSnapshot = { revision: Text, root: Text, files: EditorFile[] };
export type EditPatch = { name: Text, start: Num, end: Num, text: Text,
  expected_revision: Text };
export type EditReport = { status: Text, revision: Text, detail: Text };
export type CheckReport = { status: Text, revision: Text, source_revision: Text, detail: Text };
export type RunReport = { status: Text, run_id: Text, revision: Text,
  source_revision: Text, value_text: Text, trace_events: Num, detail: Text };
export type TraceView = { run_id: Text, source_revision: Text,
  index: Num, total: Num, event_json: Text };
export type ViewPanel = { heading: Text, body: Text };
export type EditorView = { title: Text, panels: ViewPanel[] };
