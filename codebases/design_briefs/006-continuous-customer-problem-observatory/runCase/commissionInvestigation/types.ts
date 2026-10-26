export type PreparedInvestigationBrief = { candidate: TrendSlice; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsInvestigationBrief = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditInvestigationBrief = { verdict: GateVerdict; findings: string[]; repairTarget: string };
