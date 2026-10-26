export type PreparedDisclosureScope = { candidate: DisclosureCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisclosureScope = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisclosureScope = { verdict: GateVerdict; findings: string[]; repairTarget: string };
