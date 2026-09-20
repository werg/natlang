export type Section = { id: Text; heading: Text; body: Text }; export type State = { revision: Num; notice: Text; title: Text; sections: Section[]; published: Bool };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "save"; target: Text; secondary: Text; text: Text; } | { action: "title"; text: Text; } | { action: "add"; secondary: Text; text: Text; } | { action: "reorder"; ids: Text[]; } | { action: "publish";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
