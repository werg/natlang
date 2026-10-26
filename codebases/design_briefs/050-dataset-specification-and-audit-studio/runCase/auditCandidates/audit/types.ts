export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DatasetUse } };
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
export type DatasetProject = RecordMeta & {
  intendedUse: string;
  targetPopulation: string;
  sourceCollectionIds: ID[];
  ownerId: ID;
  releaseAudience: string;
};
export type DatasetUse = RecordMeta & {
  caseId: ID;
  task: string;
  targetPopulation: string;
  prohibitedUses: string[];
  requiredSlices: string[];
};
export type AnnotationRule = RecordMeta & {
  caseId: ID;
  label: string;
  inclusionRule: string;
  exclusionRule: string;
  ambiguityPolicy: string;
};
export type DatasetRecordAudit = RecordMeta & {
  caseId: ID;
  recordId: ID;
  eligibility: string;
  qualityIssues: string[];
  rightsStatus: string;
};
export type CoverageFinding = RecordMeta & {
  caseId: ID;
  slice: string;
  eligibleCount: number;
  targetCount: number;
  gapDescription: string;
};
export type DatasetRelease = RecordMeta & {
  caseId: ID;
  acquisitionTaskIds: ID[];
  annotationVersion: string;
  artifactIds: ID[];
  unresolvedLimits: string[];
};
export type DraftDatasetUse = { values: { caseId?: ID; task?: string; targetPopulation?: string; prohibitedUses?: string[]; requiredSlices?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDatasetUse = { accepted: DatasetUse[]; unresolved: DraftDatasetUse[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftAnnotationRule = { values: { caseId?: ID; label?: string; inclusionRule?: string; exclusionRule?: string; ambiguityPolicy?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultAnnotationRule = { accepted: AnnotationRule[]; unresolved: DraftAnnotationRule[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDatasetRecordAudit = { values: { caseId?: ID; recordId?: ID; eligibility?: string; qualityIssues?: string[]; rightsStatus?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDatasetRecordAudit = { accepted: DatasetRecordAudit[]; unresolved: DraftDatasetRecordAudit[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftCoverageFinding = { values: { caseId?: ID; slice?: string; eligibleCount?: number; targetCount?: number; gapDescription?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultCoverageFinding = { accepted: CoverageFinding[]; unresolved: DraftCoverageFinding[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDatasetRelease = { values: { caseId?: ID; acquisitionTaskIds?: ID[]; annotationVersion?: string; artifactIds?: ID[]; unresolvedLimits?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDatasetRelease = { accepted: DatasetRelease[]; unresolved: DraftDatasetRelease[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; defineUseContract: StageResultDatasetUse; writeAnnotationSpec: StageResultAnnotationRule; auditCandidates: StageResultDatasetRecordAudit; measureCoverage: StageResultCoverageFinding; commissionAndRelease: StageResultDatasetRelease; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { defineUseContract: StageResultDatasetUse; writeAnnotationSpec: StageResultAnnotationRule; auditCandidates: StageResultDatasetRecordAudit; measureCoverage: StageResultCoverageFinding; commissionAndRelease: StageResultDatasetRelease; };
