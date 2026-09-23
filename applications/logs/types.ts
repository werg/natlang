export type LogEvent = { kind: "log" | "gap", id: string, cursor: number, occurred_at: number, arrived_at: number,
  service: string, code: string, level: string, message: string };
export type Observation = { id: string, status: "new" | "duplicate", service: string, code: string, occurred_at: number,
  count: number, late: boolean };
export type Evidence = { id: string, occurred_at: number, level: string, message: string };
export type Judgement = { action: "ignore" | "investigate" | "escalate", severity: string, claim: string, uncertainty: string };
