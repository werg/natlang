export type PreparedNegotiationPosition = { candidate: ClauseNegotiation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsNegotiationPosition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditNegotiationPosition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
