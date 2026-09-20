export type Merchant = { id: Text; name: Text; cash: Num; apples: Num; price: Num }; export type Trade = { id: Text; buyer: Text; seller: Text; quantity: Num; total: Num }; export type State = { revision: Num; notice: Text; merchants: Merchant[]; trades: Trade[]; tick: Num };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "price"; target: Text; amount: Num; } | { action: "buy"; target: Text; secondary: Text; amount: Num; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
