export type Receipt = { id: Text; operation: Text; status: Text; output: Text }; export type State = { revision: Num; notice: Text; asset: Text; output: Text; receipts: Receipt[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "sample";  } | { action: "import"; target: Text; } | { action: "transform"; target: Text; text: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
