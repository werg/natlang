export type PreparedCounteroffer = { candidate: SourcingPackage; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCounteroffer = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCounteroffer = { verdict: GateVerdict; findings: string[]; repairTarget: string };
