export type PreparedSubstituteCandidate = { candidate: FunctionalRequirement; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSubstituteCandidate = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSubstituteCandidate = { verdict: GateVerdict; findings: string[]; repairTarget: string };
