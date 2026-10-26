export type PreparedCommitment = { candidate: Handover; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCommitment = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCommitment = { verdict: GateVerdict; findings: string[]; repairTarget: string };
