export type PreparedMethodStep = { candidate: ReplicationProject; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMethodStep = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMethodStep = { verdict: GateVerdict; findings: string[]; repairTarget: string };
