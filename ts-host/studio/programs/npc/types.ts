export type Memory = { id: Text; text: Text }; export type Line = { id: Text; speaker: Text; text: Text; evidence: Text[] }; export type State = { revision: Num; notice: Text; memories: Memory[]; lines: Line[]; trust: Num };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "say"; text: Text; } | { action: "reply"; text: Text; ids?: Text[]; amount?: Num; } | { action: "remember"; text: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
