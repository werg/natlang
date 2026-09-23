export type Entry = { id: string; replica: string; text: string }; export type State = { revision: number; notice: string; family: string; base: string; left: string; right: string; merged: string; issues: string[]; history: Entry[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "configure"; target: string; text: string; secondary: string; } | { action: "right"; text: string; } | { action: "propose"; text: string; ids?: string[]; } | { action: "commit";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
