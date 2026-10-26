export type PreparedGrantSubmission = { candidate: GrantNarrative; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsGrantSubmission = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditGrantSubmission = { verdict: GateVerdict; findings: string[]; repairTarget: string };
