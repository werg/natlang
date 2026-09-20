export type Spell = { id: Text; words: Text; element: Text; power: Num; cost: Num }; export type State = { revision: Num; notice: Text; mana: Num; targetHp: Num; ward: Text; spells: Spell[]; last: Text };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "cast"; target: Text; text: Text; amount: Num; } | { action: "rest";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
