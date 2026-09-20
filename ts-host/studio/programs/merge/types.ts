export type Entry = { id: Text; replica: Text; text: Text }; export type State = { revision: Num; notice: Text; family: Text; base: Text; left: Text; right: Text; merged: Text; issues: Text[]; history: Entry[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "configure"; target: Text; text: Text; secondary: Text; } | { action: "right"; text: Text; } | { action: "propose"; text: Text; ids?: Text[]; } | { action: "commit";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
