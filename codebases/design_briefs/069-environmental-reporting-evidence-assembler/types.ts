export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type EnvironmentalReport = RecordMeta & {
  organizationId: ID;
  reportingPeriod: string;
  requirementIds: ID[];
  reviewerId: ID;
  boundaryDefinition: string;
};
export type ReportingBoundary = RecordMeta & {
  caseId: ID;
  includedOperations: ID[];
  excludedOperations: ID[];
  measurementRules: string[];
  effectiveScope: string;
};
export type EnvironmentalMeasure = RecordMeta & {
  caseId: ID;
  operationId: ID;
  measureKind: string;
  value: Decimal;
  unit: string;
  methodRef: Ref;
};
export type EnvironmentalAllocation = RecordMeta & {
  caseId: ID;
  measureIds: ID[];
  boundaryId: ID;
  allocatedValue: Decimal;
  allocationBasis: string;
};
export type EnvironmentalClaim = RecordMeta & {
  caseId: ID;
  statement: string;
  measureIds: ID[];
  activityRefs: Ref[];
  limitation: string;
};
export type EnvironmentalSubmission = RecordMeta & {
  caseId: ID;
  claimIds: ID[];
  scheduleArtifactIds: ID[];
  unresolvedGaps: string[];
  approvalRefs: Ref[];
};
export type DraftReportingBoundary = { values: { caseId?: ID; includedOperations?: ID[]; excludedOperations?: ID[]; measurementRules?: string[]; effectiveScope?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type ReportingOfficer = { id: string };
export type ReportingDossier = { report: EnvironmentalReport; boundary: ReportingBoundary; measures: EnvironmentalMeasure[]; allocations: EnvironmentalAllocation[]; claims: EnvironmentalClaim[]; submissions: EnvironmentalSubmission[]; activityEvidence: Evidence[]; duplicateActivityIds: string[]; checkpoint: ReportCheckpoint };
export type ReportChange = { newEvidence: Evidence[]; changedBoundary: ReportingBoundary | null; proposedClaims: EnvironmentalClaim[]; duplicateActivityCandidates: string[] };
export type ActivityIdentity = { activityId: string; sourceRefs: string[]; measureIds: string[]; relation: 'same-activity'|'distinct-activity'|'unresolved'; basis: string; };
export type ReconciliationLedger = { identities: ActivityIdentity[]; allocations: EnvironmentalAllocation[]; exactTotals: Record<string, Decimal>; unresolved: string[]; };
export type ClaimAssessment = { claimId: string; support: 'substantiated'|'qualified'|'unsupported'; measureIds: string[]; limitation: string; findings: string[]; };
export type ReportCheckpoint = { edition: string; boundary: ReportingBoundary; measures: EnvironmentalMeasure[]; allocations: EnvironmentalAllocation[]; claims: EnvironmentalClaim[]; submissions: EnvironmentalSubmission[]; restatementBridge: string[]; invalidatedClaimIds: string[]; };
export type ReportingProposal = { application: 'Environmental reporting evidence assembler'; status: 'ready-for-review'|'blocked'; boundary: ReportingBoundary; measures: EnvironmentalMeasure[]; reconciliation: ReconciliationLedger; claimAssessments: ClaimAssessment[]; submission: EnvironmentalSubmission; restatement: string[]; blockers: string[]; actionBoundary: string; resumeState: ReportCheckpoint; };
