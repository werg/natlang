export type WorkflowStep = { id: Text; title: Text; needs: Text[]; status: Text; key: Text }; export type Receipt = { id: Text; step: Text; status: Text }; export type State = { revision: Num; notice: Text; steps: WorkflowStep[]; receipts: Receipt[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "execute"; target: Text; } | { action: "fail"; target: Text; } | { action: "unknown"; target: Text; } | { action: "reconcile"; target: Text; text: Text; } | { action: "compensate"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
