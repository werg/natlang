export type PreparedResearchLesson = { candidate: DiscriminatingExperiment; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResearchLesson = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResearchLesson = { verdict: GateVerdict; findings: string[]; repairTarget: string };
