export type PreparedObligationStatus = { candidate: ObligationTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsObligationStatus = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditObligationStatus = { verdict: GateVerdict; findings: string[]; repairTarget: string };
