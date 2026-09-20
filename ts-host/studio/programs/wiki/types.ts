export type Page = { id: Text; title: Text; body: Text; cell: Text; output: Text }; export type State = { revision: Num; notice: Text; pages: Page[]; selected: Text; incoming: Text; conflicts: Text[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "save"; secondary: Text; text: Text; } | { action: "add"; target: Text; secondary: Text; text: Text; } | { action: "select"; target: Text; } | { action: "incoming"; text: Text; } | { action: "merge"; text: Text; ids?: Text[]; } | { action: "save_cell"; text: Text; } | { action: "run"; text?: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
