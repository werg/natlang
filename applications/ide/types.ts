export type EditorFile = { name: string, source: string };
export type EditorSnapshot = { revision: string, root: string, files: EditorFile[] };
export type EditPatch = { name: string, start: number, end: number, text: string, expected_revision: string };
export type CheckReport = { status: "checked" | "invalid", revision: string, detail: string };
export type TraceView = { run_id: string, revision: string, index: number, total: number, event_json: string };
export type ViewPanel = { heading: string, body: string };
export type EditorView = { title: string, panels: ViewPanel[] };
