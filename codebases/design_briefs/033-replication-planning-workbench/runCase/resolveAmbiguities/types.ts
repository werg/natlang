export type PreparedMethodQuestion = { candidate: ReplicationDependency; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMethodQuestion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMethodQuestion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
