export type PreparedActivityEvidence = { candidate: GrantAllocation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsActivityEvidence = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditActivityEvidence = { verdict: GateVerdict; findings: string[]; repairTarget: string };
