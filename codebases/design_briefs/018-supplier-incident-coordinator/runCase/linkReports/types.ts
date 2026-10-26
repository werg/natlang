export type PreparedIncidentCluster = { candidate: SupplierIncident; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsIncidentCluster = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditIncidentCluster = { verdict: GateVerdict; findings: string[]; repairTarget: string };
