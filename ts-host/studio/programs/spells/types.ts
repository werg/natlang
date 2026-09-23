export type Spell = { id: string; words: string; element: string; power: number; cost: number }; export type State = { revision: number; notice: string; mana: number; targetHp: number; ward: string; spells: Spell[]; last: string };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "cast"; target: string; text: string; amount: number; } | { action: "rest";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
