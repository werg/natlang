export type PreparedGrantAllocation = { candidate: GrantCondition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsGrantAllocation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditGrantAllocation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
