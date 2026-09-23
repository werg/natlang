export type WorkflowStep = { id: string; title: string; needs: string[]; status: string; key: string }; export type Receipt = { id: string; step: string; status: string }; export type State = { revision: number; notice: string; steps: WorkflowStep[]; receipts: Receipt[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "execute"; target: string; } | { action: "fail"; target: string; } | { action: "unknown"; target: string; } | { action: "reconcile"; target: string; text: string; } | { action: "compensate"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
