export type PreparedReplicationDependency = { candidate: MethodStep; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReplicationDependency = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReplicationDependency = { verdict: GateVerdict; findings: string[]; repairTarget: string };
