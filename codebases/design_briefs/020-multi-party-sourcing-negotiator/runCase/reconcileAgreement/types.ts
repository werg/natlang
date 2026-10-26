export type PreparedSourcingAgreement = { candidate: Counteroffer; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSourcingAgreement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSourcingAgreement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
