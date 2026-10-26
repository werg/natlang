export type PreparedWorkStatus = { candidate: Commitment; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsWorkStatus = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditWorkStatus = { verdict: GateVerdict; findings: string[]; repairTarget: string };
