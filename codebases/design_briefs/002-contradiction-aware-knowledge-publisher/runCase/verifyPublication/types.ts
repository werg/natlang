export type PreparedPublicationCheck = { candidate: PagePatch; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPublicationCheck = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPublicationCheck = { verdict: GateVerdict; findings: string[]; repairTarget: string };
