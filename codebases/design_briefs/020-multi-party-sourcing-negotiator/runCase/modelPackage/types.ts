export type PreparedSourcingRequirement = { candidate: SourcingEvent; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSourcingRequirement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSourcingRequirement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
