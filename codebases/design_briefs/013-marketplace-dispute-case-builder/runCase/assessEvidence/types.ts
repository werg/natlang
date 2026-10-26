export type PreparedEvidenceFinding = { candidate: DisputedQuestion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEvidenceFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEvidenceFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
