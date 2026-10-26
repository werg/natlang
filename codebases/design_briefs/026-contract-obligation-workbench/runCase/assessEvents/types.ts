export type PreparedTriggerFinding = { candidate: ObligationRelation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsTriggerFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditTriggerFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
