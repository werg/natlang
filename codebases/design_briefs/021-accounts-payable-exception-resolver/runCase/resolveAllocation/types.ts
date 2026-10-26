export type PreparedAllocationProposal = { candidate: PayableDiscrepancy; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAllocationProposal = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAllocationProposal = { verdict: GateVerdict; findings: string[]; repairTarget: string };
