export type PreparedDisclosureSelection = { candidate: DisclosureScope; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisclosureSelection = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisclosureSelection = { verdict: GateVerdict; findings: string[]; repairTarget: string };
