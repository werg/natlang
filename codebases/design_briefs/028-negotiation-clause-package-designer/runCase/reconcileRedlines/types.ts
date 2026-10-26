export type PreparedRedlineFinding = { candidate: ClausePackage; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRedlineFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRedlineFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
