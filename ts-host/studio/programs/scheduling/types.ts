export type Task = { id: Text; title: Text; duration: Num; start: Text; done: Bool }; export type State = { revision: Num; notice: Text; tasks: Task[]; date: Text };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "add"; text: Text; amount: Num; } | { action: "date"; text: Text; } | { action: "schedule"; target: Text; text: Text; } | { action: "complete"; target: Text; } | { action: "unschedule"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
