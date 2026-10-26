export type PreparedContinuityReview = { candidate: OwnershipTransfer; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsContinuityReview = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditContinuityReview = { verdict: GateVerdict; findings: string[]; repairTarget: string };
