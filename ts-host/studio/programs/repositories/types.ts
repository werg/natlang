export type Receipt = { id: Text; operation: Text; status: Text; output: Text }; export type State = { revision: Num; notice: Text; before: Text; after: Text; receipts: Receipt[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "propose"; text: Text; secondary: Text; } | { action: "check";  } | { action: "reset";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
