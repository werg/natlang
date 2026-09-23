export type Receipt = { id: string; operation: string; status: string; output: string }; export type State = { revision: number; notice: string; catalog: string; lock: string; receipts: Receipt[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "browse";  } | { action: "resolve"; target: string; text: string; } | { action: "install"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
