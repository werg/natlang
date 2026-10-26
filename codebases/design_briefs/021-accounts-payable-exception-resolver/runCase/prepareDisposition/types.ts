export type PreparedPayableDisposition = { candidate: AllocationProposal; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPayableDisposition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPayableDisposition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
