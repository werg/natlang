export type PreparedExperimentDesign = { candidate: ExperimentReview; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExperimentDesign = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExperimentDesign = { verdict: GateVerdict; findings: string[]; repairTarget: string };
