export type PreparedGrantCondition = { candidate: GrantReport; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsGrantCondition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditGrantCondition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
