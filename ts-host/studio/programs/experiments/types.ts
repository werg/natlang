export type Trial = { id: Text; policy: Text; seed: Num; reward: Num; steps: Num }; export type State = { revision: Num; notice: Text; trials: Trial[]; hypothesis: Text };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "hypothesis"; text: Text; } | { action: "run"; target: Text; amount: Num; count: Num; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
