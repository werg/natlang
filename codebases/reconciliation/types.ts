export type Customer = { id: string, tier: string };
export type Event = { id: string, customer: string, message: string, cents: number };
export type Joined = { id: string, customer: string, message: string, cents: number, matched: boolean, tier: string };
export type Join = { rows: Joined[], duplicates: number };
export type Label = "urgent" | "normal";
export type Report = { totals: Record<string, number>, urgent: string[], unmatched: string[], duplicates: number };
