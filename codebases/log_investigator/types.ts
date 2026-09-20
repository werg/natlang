export type LogEvent = { kind: Text, id: Text, cursor: Num, occurred_at: Num, arrived_at: Num, service: Text, code: Text, level: Text, message: Text };
export type Observation = { id: Text, status: Text, service: Text, code: Text, occurred_at: Num, count: Num, late: Bool };
export type Evidence = { id: Text, occurred_at: Num, level: Text, message: Text };
export type Judgement = { action: Text, severity: Text, claim: Text, uncertainty: Text };
export type Alert = { status: Text, key: Text, detail: Text };
export type IncidentState = { cursor: Num, observed: Num, alerts: Alert[], unknowns: Text[], status: Text };
