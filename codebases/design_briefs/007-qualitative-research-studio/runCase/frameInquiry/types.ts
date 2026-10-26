export type PreparedInquiryFrame = { candidate: ResearchProject; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsInquiryFrame = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditInquiryFrame = { verdict: GateVerdict; findings: string[]; repairTarget: string };
