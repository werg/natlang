export type Page = { id: string; title: string; body: string; cell: string; output: string }; export type State = { revision: number; notice: string; pages: Page[]; selected: string; incoming: string; conflicts: string[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "save"; secondary: string; text: string; } | { action: "add"; target: string; secondary: string; text: string; } | { action: "select"; target: string; } | { action: "incoming"; text: string; } | { action: "merge"; text: string; ids?: string[]; } | { action: "save_cell"; text: string; } | { action: "run"; text?: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
