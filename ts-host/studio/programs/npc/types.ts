export type Memory = { id: string; text: string }; export type Line = { id: string; speaker: string; text: string; evidence: string[] }; export type State = { revision: number; notice: string; memories: Memory[]; lines: Line[]; trust: number };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "say"; text: string; } | { action: "reply"; text: string; ids?: string[]; amount?: number; } | { action: "remember"; text: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
