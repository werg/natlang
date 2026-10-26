export type PreparedIntervention = { candidate: CustomerNeed; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsIntervention = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditIntervention = { verdict: GateVerdict; findings: string[]; repairTarget: string };
