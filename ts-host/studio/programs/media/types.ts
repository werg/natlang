export type Receipt = { id: string; operation: string; status: string; output: string }; export type State = { revision: number; notice: string; asset: string; output: string; receipts: Receipt[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "sample";  } | { action: "import"; target: string; } | { action: "transform"; target: string; text: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
