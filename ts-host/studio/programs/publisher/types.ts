export type Section = { id: string; heading: string; body: string }; export type State = { revision: number; notice: string; title: string; sections: Section[]; published: boolean };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "save"; target: string; secondary: string; text: string; } | { action: "title"; text: string; } | { action: "add"; secondary: string; text: string; } | { action: "reorder"; ids: string[]; } | { action: "publish";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
