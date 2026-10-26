export type PreparedReplicationFinding = { candidate: ReplicationTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReplicationFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReplicationFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
