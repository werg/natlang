export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: ResponseObservation } };
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
export type LearningCase = RecordMeta & {
  learnerId: ID;
  competencyIds: ID[];
  learningGoal: string;
  instructorId: ID;
  accommodationIds: ID[];
};
export type ResponseObservation = RecordMeta & {
  caseId: ID;
  taskId: ID;
  responseRef: Ref;
  observedStrategy: string;
  errorLocation: string;
};
export type MisconceptionHypothesis = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  hypothesis: string;
  rivalHypotheses: string[];
  discriminatingFeature: string;
};
export type DiagnosticTask = RecordMeta & {
  caseId: ID;
  hypothesisIds: ID[];
  prompt: string;
  expectedResponseClasses: string[];
  prerequisiteIds: ID[];
};
export type LearningIntervention = RecordMeta & {
  caseId: ID;
  hypothesisId: ID;
  explanation: string;
  workedExample: string;
  practiceTaskIds: ID[];
};
export type TransferFinding = RecordMeta & {
  caseId: ID;
  competencyId: ID;
  novelTaskIds: ID[];
  demonstratedSkill: string;
  remainingUncertainty: string;
};
export type DraftResponseObservation = { values: { caseId?: ID; taskId?: ID; responseRef?: Ref; observedStrategy?: string; errorLocation?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultResponseObservation = { accepted: ResponseObservation[]; unresolved: DraftResponseObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMisconceptionHypothesis = { values: { caseId?: ID; observationIds?: ID[]; hypothesis?: string; rivalHypotheses?: string[]; discriminatingFeature?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMisconceptionHypothesis = { accepted: MisconceptionHypothesis[]; unresolved: DraftMisconceptionHypothesis[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDiagnosticTask = { values: { caseId?: ID; hypothesisIds?: ID[]; prompt?: string; expectedResponseClasses?: string[]; prerequisiteIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDiagnosticTask = { accepted: DiagnosticTask[]; unresolved: DraftDiagnosticTask[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftLearningIntervention = { values: { caseId?: ID; hypothesisId?: ID; explanation?: string; workedExample?: string; practiceTaskIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultLearningIntervention = { accepted: LearningIntervention[]; unresolved: DraftLearningIntervention[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftTransferFinding = { values: { caseId?: ID; competencyId?: ID; novelTaskIds?: ID[]; demonstratedSkill?: string; remainingUncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultTransferFinding = { accepted: TransferFinding[]; unresolved: DraftTransferFinding[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretResponses: StageResultResponseObservation; formHypotheses: StageResultMisconceptionHypothesis; generateDiagnostics: StageResultDiagnosticTask; planRepair: StageResultLearningIntervention; verifyTransfer: StageResultTransferFinding; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretResponses: StageResultResponseObservation; formHypotheses: StageResultMisconceptionHypothesis; generateDiagnostics: StageResultDiagnosticTask; planRepair: StageResultLearningIntervention; verifyTransfer: StageResultTransferFinding; };
export type TaskQualityReview = { taskId: ID; clear: boolean; ambiguitySources: string[]; accessibilityBarriers: string[]; competencyPreserved: string; repairQuestions: Question[]; sourceRefs: Ref[] };
export type LearnerEvidence = { taskId: ID; responseRef: Ref; acceptedStrategyIds: string[]; demonstratedSteps: string[]; errorClass: 'conceptual' | 'arithmetic' | 'reading' | 'incomplete' | 'unresolved'; confidence: 'supported' | 'tentative' | 'unknown'; hintLevel: number; sourceRefs: Ref[] };
export type HypothesisSet = { caseId: ID; candidates: MisconceptionHypothesis[]; ambiguityRivalPresent: boolean; evidenceNeeded: string[]; diagnosticQuestion: Question[] };
export type UnbiasedDiagnostic = { task: DiagnosticTask; changedFeature: string; heldConstant: string[]; hiddenLabels: string[]; independentlyCheckedAnswerSpace: string[]; accessReviewRefs: Ref[] };
export type LearnerCheckpoint = { caseId: ID; lastCompletedTaskIds: ID[]; privateHypothesisIds: ID[]; learnerVisibleText: string; accommodationIds: ID[]; hintExposureByTask: string[]; handoffAttemptCount: number; pendingQuestionIds: ID[] };
export type BoundedHandoff = { offered: boolean; reason: string; instructorId: ID; supportingRefs: Ref[]; unresolvedSinceCheckpoint: string[] };
export type LearningRepairSession = { caseId: ID; taskQuality: TaskQualityReview[]; observations: LearnerEvidence[]; hypotheses: HypothesisSet[]; diagnostics: UnbiasedDiagnostic[]; interventions: LearningIntervention[]; transfer: TransferFinding[]; checkpoint: LearnerCheckpoint; instructorHandoff: BoundedHandoff; hiddenLabelsExposed: boolean; masteryClaim: boolean };
export type LearnerSessionInput = { caseId: ID; learningGoal: string; sourceEvidence: Evidence[]; authorizedPolicyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; changedEvidenceIds: ID[]; learnerAudience: string; observedResponse: ResponseObservation; attemptLimit?: number };
