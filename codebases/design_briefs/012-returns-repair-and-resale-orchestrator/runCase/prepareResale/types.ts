export type PreparedResaleDisclosure = { candidate: CustodyEvent; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResaleDisclosure = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResaleDisclosure = { verdict: GateVerdict; findings: string[]; repairTarget: string };
