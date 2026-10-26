export type PreparedDisputedQuestion = { candidate: DisputeEvent; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisputedQuestion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisputedQuestion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
