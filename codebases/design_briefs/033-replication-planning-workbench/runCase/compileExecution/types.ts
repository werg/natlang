export type PreparedReplicationTask = { candidate: MethodQuestion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReplicationTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReplicationTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
