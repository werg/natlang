export type Task = { id: string; title: string; duration: number; start: string; done: boolean }; export type State = { revision: number; notice: string; tasks: Task[]; date: string };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "add"; text: string; amount: number; } | { action: "date"; text: string; } | { action: "schedule"; target: string; text: string; } | { action: "complete"; target: string; } | { action: "unschedule"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
