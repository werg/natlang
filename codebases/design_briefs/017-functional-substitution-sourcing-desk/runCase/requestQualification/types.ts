export type PreparedQualificationTask = { candidate: CompatibilityFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsQualificationTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditQualificationTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
