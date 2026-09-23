export type Severity = "low" | "high";
export type Decision = { action: "allow" | "warn" | "remove" | "escalate", rules: string[], note: string };
