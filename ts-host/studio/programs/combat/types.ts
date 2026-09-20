export type Fighter = { id: Text; hp: Num; energy: Num; stance: Text }; export type Round = { id: Text; player: Text; rival: Text; detail: Text }; export type State = { revision: Num; notice: Text; fighters: Fighter[]; rounds: Round[]; seed: Num };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "round"; text: Text; secondary: Text; } | { action: "reset";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
