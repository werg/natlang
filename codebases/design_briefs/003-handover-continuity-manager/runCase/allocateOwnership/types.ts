export type PreparedOwnershipTransfer = { candidate: HandoverDependency; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOwnershipTransfer = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOwnershipTransfer = { verdict: GateVerdict; findings: string[]; repairTarget: string };
