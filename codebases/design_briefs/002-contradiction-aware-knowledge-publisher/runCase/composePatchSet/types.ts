export type PreparedPagePatch = { candidate: AuthorityDecision; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPagePatch = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPagePatch = { verdict: GateVerdict; findings: string[]; repairTarget: string };
