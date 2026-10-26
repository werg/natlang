export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: ReportingBoundary } };
export type GoalContract = { outcome: string; exclusions: string[]; affectedEntities: string[]; acceptanceEvidence: string[]; clarificationNeeded: string[] };
export type PolicySelection = { selected: string[]; unresolvedPrecedence: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized" };
export type WorkGraph = { nodes: string[]; prerequisites: string[]; blockedQuestions: string[] };
export type EvidencePlan = { requiredEvidence: string[]; selectedIds: string[]; searchedUniverse: string[]; missingCoverage: string[] };
export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";
export type AuditResult = { verdict: Qualification; gate: string; reason: string; repairTargets: string[] };
export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };
export type Rejection = { id: string; reason: string; evidenceIds: string[] };
export type CoverageManifest = { eligible: string[]; processed: string[]; deferred: string[]; rejected: string[]; outOfScope: string[] };
export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };
export type ActionProposal = { kind: string; targetIds: string[]; reason: string; approvalRequired: boolean };
export type CheckResult = { check: string; verdict: Qualification; evidenceIds: string[] };
export type CollectionFinding = { acceptedRefs: string[]; unresolvedRefs: string[]; conflicts: string[]; coverageGaps: string[] };
export type QuestionBatch = { questions: string[]; authorizedRespondent: string; acceptableEvidence: string[] };
export type InvalidationProposal = { invalidated: string[]; preserved: string[]; unresolvedImpact: string[] };
export type ReviewPacket = { supportedDecisions: string[]; unresolvedAlternatives: string[]; actionScope: string[]; approvalBindings: string[] };
export type ViewModel = { title: string; status: "ready" | "blocked" | "review"; sections: string[]; uncertainties: string[] };
export type RevisionProposal = { target: string; change: string; contracts: string[]; expectedBenefit: string; regressionRisks: string[]; tests: string[] };

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
export type StageResultReportingBoundary = { accepted: ReportingBoundary[]; unresolved: DraftReportingBoundary[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftEnvironmentalMeasure = { values: { caseId?: ID; operationId?: ID; measureKind?: string; value?: Decimal; unit?: string; methodRef?: Ref }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultEnvironmentalMeasure = { accepted: EnvironmentalMeasure[]; unresolved: DraftEnvironmentalMeasure[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftEnvironmentalAllocation = { values: { caseId?: ID; measureIds?: ID[]; boundaryId?: ID; allocatedValue?: Decimal; allocationBasis?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultEnvironmentalAllocation = { accepted: EnvironmentalAllocation[]; unresolved: DraftEnvironmentalAllocation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftEnvironmentalClaim = { values: { caseId?: ID; statement?: string; measureIds?: ID[]; activityRefs?: Ref[]; limitation?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultEnvironmentalClaim = { accepted: EnvironmentalClaim[]; unresolved: DraftEnvironmentalClaim[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftEnvironmentalSubmission = { values: { caseId?: ID; claimIds?: ID[]; scheduleArtifactIds?: ID[]; unresolvedGaps?: string[]; approvalRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultEnvironmentalSubmission = { accepted: EnvironmentalSubmission[]; unresolved: DraftEnvironmentalSubmission[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretBoundaries: StageResultReportingBoundary; mapMeasurements: StageResultEnvironmentalMeasure; reconcileAttribution: StageResultEnvironmentalAllocation; substantiateClaims: StageResultEnvironmentalClaim; assembleSubmission: StageResultEnvironmentalSubmission; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretBoundaries: StageResultReportingBoundary; mapMeasurements: StageResultEnvironmentalMeasure; reconcileAttribution: StageResultEnvironmentalAllocation; substantiateClaims: StageResultEnvironmentalClaim; assembleSubmission: StageResultEnvironmentalSubmission; };
