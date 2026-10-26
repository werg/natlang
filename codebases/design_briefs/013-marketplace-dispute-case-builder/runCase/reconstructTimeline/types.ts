export type PreparedDisputeEvent = { candidate: DisputeCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisputeEvent = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisputeEvent = { verdict: GateVerdict; findings: string[]; repairTarget: string };
