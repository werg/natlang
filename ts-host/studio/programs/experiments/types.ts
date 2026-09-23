export type Trial = { id: string; policy: string; seed: number; reward: number; steps: number }; export type State = { revision: number; notice: string; trials: Trial[]; hypothesis: string };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "hypothesis"; text: string; } | { action: "run"; target: string; amount: number; count: number; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
