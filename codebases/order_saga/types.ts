export type Event = { id: string, order: string, text: string };
export type Kind = "start" | "paid" | "failed" | "sent" | "cancel";
export type Order = "reserved" | "paid" | "done" | "cancelled";
export type Command = { key: string, order: string, operation: string };
export type State = { seen: string[], orders: Record<string, Order>, outbox: Command[] };
