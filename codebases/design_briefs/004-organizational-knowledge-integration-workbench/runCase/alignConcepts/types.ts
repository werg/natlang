export type PreparedConceptAlignment = { candidate: TermSense; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsConceptAlignment = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditConceptAlignment = { verdict: GateVerdict; findings: string[]; repairTarget: string };
