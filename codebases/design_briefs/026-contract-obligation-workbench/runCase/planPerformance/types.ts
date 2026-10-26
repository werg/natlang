export type PreparedObligationTask = { candidate: TriggerFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsObligationTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditObligationTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
