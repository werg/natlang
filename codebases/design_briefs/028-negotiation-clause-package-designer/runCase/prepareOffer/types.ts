export type PreparedClauseOffer = { candidate: RedlineFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsClauseOffer = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditClauseOffer = { verdict: GateVerdict; findings: string[]; repairTarget: string };
