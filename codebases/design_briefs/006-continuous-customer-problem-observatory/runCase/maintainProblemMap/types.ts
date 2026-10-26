export type PreparedProblemConcept = { candidate: IncidentGroup; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsProblemConcept = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditProblemConcept = { verdict: GateVerdict; findings: string[]; repairTarget: string };
