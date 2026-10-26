export type PreparedHandoverDependency = { candidate: WorkStatus; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsHandoverDependency = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditHandoverDependency = { verdict: GateVerdict; findings: string[]; repairTarget: string };
