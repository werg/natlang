export type PreparedProblemEpisode = { candidate: ProblemStudy; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsProblemEpisode = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditProblemEpisode = { verdict: GateVerdict; findings: string[]; repairTarget: string };
