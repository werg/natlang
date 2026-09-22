type Document = { revision: number, text: string };
type Update = { id: string, parents: string[], base_revision: number, author: string, text: string };
type Alternative = { update_ids: string[], proposal: string, reason: string };
type Draft = { text: string, applied: string[], alternatives: Alternative[], explanation: string };
type Prepared = { valid: boolean, error: string, updates: Update[], presentation: string };
type MergeResult = { status: "merged" | "unresolved" | "rejected", text: string, applied: string[], alternatives: Alternative[], explanation: string, base_revision: number, updates: Update[], presentation: string };
type Step = { kind: "new" | "duplicate" | "invalid", error: string, updates: Update[], presentation: string };
