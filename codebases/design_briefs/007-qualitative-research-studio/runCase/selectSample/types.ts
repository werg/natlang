export type PreparedSamplingPlan = { candidate: InquiryFrame; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSamplingPlan = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSamplingPlan = { verdict: GateVerdict; findings: string[]; repairTarget: string };
