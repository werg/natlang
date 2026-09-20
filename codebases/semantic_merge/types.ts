type Document = { revision: Num, text: Text };
type Update = { id: Text, parents: Text[], base_revision: Num, author: Text, text: Text };
type Alternative = { update_ids: Text[], proposal: Text, reason: Text };
type Draft = { text: Text, applied: Text[], alternatives: Alternative[], explanation: Text };
type Prepared = { valid: Bool, error: Text, updates: Update[], presentation: Text };
type MergeResult = { status: "merged" | "unresolved" | "rejected", text: Text, applied: Text[], alternatives: Alternative[], explanation: Text, base_revision: Num, updates: Update[], presentation: Text };
type Step = { kind: "new" | "duplicate" | "invalid", error: Text, updates: Update[], presentation: Text };
