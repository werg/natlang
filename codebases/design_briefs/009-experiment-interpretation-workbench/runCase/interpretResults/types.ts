export type PreparedResultClaim = { candidate: MeasurementFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResultClaim = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResultClaim = { verdict: GateVerdict; findings: string[]; repairTarget: string };
