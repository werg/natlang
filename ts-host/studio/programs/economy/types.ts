export type Merchant = { id: string; name: string; cash: number; apples: number; price: number }; export type Trade = { id: string; buyer: string; seller: string; quantity: number; total: number }; export type State = { revision: number; notice: string; merchants: Merchant[]; trades: Trade[]; tick: number };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "price"; target: string; amount: number; } | { action: "buy"; target: string; secondary: string; amount: number; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
