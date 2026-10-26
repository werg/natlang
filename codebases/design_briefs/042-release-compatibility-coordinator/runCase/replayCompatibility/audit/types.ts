export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: BehavioralChange } };
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
export type ReleaseReview = RecordMeta & {
  releaseId: ID;
  baselineRevision: string;
  candidateRevision: string;
  integrationIds: ID[];
  releaseOwnerId: ID;
};
export type BehavioralChange = RecordMeta & {
  caseId: ID;
  componentId: ID;
  oldBehavior: string;
  newBehavior: string;
  affectedContract: string;
};
export type ConsumerDependency = RecordMeta & {
  caseId: ID;
  consumerId: ID;
  reliedOnBehavior: string;
  evidenceRefs: Ref[];
  exposure: string;
};
export type CompatibilityTest = RecordMeta & {
  caseId: ID;
  consumerId: ID;
  scenario: string;
  baselineResult: string;
  candidateResult: string;
};
export type MigrationWave = RecordMeta & {
  caseId: ID;
  consumerIds: ID[];
  prerequisiteIds: ID[];
  fallbackPlan: string;
  acceptanceCriteria: string[];
};
export type ReleaseReadiness = RecordMeta & {
  caseId: ID;
  waveIds: ID[];
  documentationIds: ID[];
  ownerApprovals: Ref[];
  unresolvedRisk: string;
};
export type DraftBehavioralChange = { values: { caseId?: ID; componentId?: ID; oldBehavior?: string; newBehavior?: string; affectedContract?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultBehavioralChange = { accepted: BehavioralChange[]; unresolved: DraftBehavioralChange[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftConsumerDependency = { values: { caseId?: ID; consumerId?: ID; reliedOnBehavior?: string; evidenceRefs?: Ref[]; exposure?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultConsumerDependency = { accepted: ConsumerDependency[]; unresolved: DraftConsumerDependency[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftCompatibilityTest = { values: { caseId?: ID; consumerId?: ID; scenario?: string; baselineResult?: string; candidateResult?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultCompatibilityTest = { accepted: CompatibilityTest[]; unresolved: DraftCompatibilityTest[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMigrationWave = { values: { caseId?: ID; consumerIds?: ID[]; prerequisiteIds?: ID[]; fallbackPlan?: string; acceptanceCriteria?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMigrationWave = { accepted: MigrationWave[]; unresolved: DraftMigrationWave[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReleaseReadiness = { values: { caseId?: ID; waveIds?: ID[]; documentationIds?: ID[]; ownerApprovals?: Ref[]; unresolvedRisk?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReleaseReadiness = { accepted: ReleaseReadiness[]; unresolved: DraftReleaseReadiness[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretChanges: StageResultBehavioralChange; mapConsumers: StageResultConsumerDependency; replayCompatibility: StageResultCompatibilityTest; planMigration: StageResultMigrationWave; coordinateRelease: StageResultReleaseReadiness; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretChanges: StageResultBehavioralChange; mapConsumers: StageResultConsumerDependency; replayCompatibility: StageResultCompatibilityTest; planMigration: StageResultMigrationWave; coordinateRelease: StageResultReleaseReadiness; };
