export type Fighter = { id: string; hp: number; energy: number; stance: string }; export type Round = { id: string; player: string; rival: string; detail: string }; export type State = { revision: number; notice: string; fighters: Fighter[]; rounds: Round[]; seed: number };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "round"; text: string; secondary: string; } | { action: "reset";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
