export type Mapping = { id: string; source: string; target: string }; export type State = { revision: number; notice: string; input: string; mappings: Mapping[]; output: string; issues: string[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "input"; text: string; } | { action: "map"; text: string; secondary: string; } | { action: "remove"; target: string; } | { action: "transform";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
