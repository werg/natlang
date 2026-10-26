export type PreparedBillingPacket = { candidate: BillableAllocation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsBillingPacket = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditBillingPacket = { verdict: GateVerdict; findings: string[]; repairTarget: string };
