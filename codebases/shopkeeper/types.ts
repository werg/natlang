export type Message = { from: string, text: string };
export type Shop = { stock: Record<string, number>, prices: Record<string, number>, coins: number, persona: string, ledger: string[] };
export type Intent = { kind: "buy" | "ask_price" | "haggle" | "chat", good: string, qty: number, offer: number };
export type Action = { code: string, good: string, qty: number, price: number };
