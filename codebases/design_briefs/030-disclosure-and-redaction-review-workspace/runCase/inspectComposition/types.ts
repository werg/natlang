export type PreparedDisclosureRisk = { candidate: RedactionProposal; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisclosureRisk = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisclosureRisk = { verdict: GateVerdict; findings: string[]; repairTarget: string };
