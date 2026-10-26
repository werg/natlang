export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type TaskQualityReview = { taskId: ID; clear: boolean; ambiguitySources: string[]; accessibilityBarriers: string[]; competencyPreserved: string; repairQuestions: Question[]; sourceRefs: Ref[] };
export type LearnerEvidence = { taskId: ID; responseRef: Ref; acceptedStrategyIds: string[]; demonstratedSteps: string[]; errorClass: 'conceptual' | 'arithmetic' | 'reading' | 'incomplete' | 'unresolved'; confidence: 'supported' | 'tentative' | 'unknown'; hintLevel: number; sourceRefs: Ref[] };
export type HypothesisSet = { caseId: ID; candidates: MisconceptionHypothesis[]; ambiguityRivalPresent: boolean; evidenceNeeded: string[]; diagnosticQuestion: Question[] };
export type UnbiasedDiagnostic = { task: DiagnosticTask; changedFeature: string; heldConstant: string[]; hiddenLabels: string[]; independentlyCheckedAnswerSpace: string[]; accessReviewRefs: Ref[] };
export type LearnerCheckpoint = { caseId: ID; lastCompletedTaskIds: ID[]; privateHypothesisIds: ID[]; learnerVisibleText: string; accommodationIds: ID[]; hintExposureByTask: string[]; handoffAttemptCount: number; pendingQuestionIds: ID[] };
export type BoundedHandoff = { offered: boolean; reason: string; instructorId: ID; supportingRefs: Ref[]; unresolvedSinceCheckpoint: string[] };
export type LearningRepairSession = { caseId: ID; taskQuality: TaskQualityReview[]; observations: LearnerEvidence[]; hypotheses: HypothesisSet[]; diagnostics: UnbiasedDiagnostic[]; interventions: LearningIntervention[]; transfer: TransferFinding[]; checkpoint: LearnerCheckpoint; instructorHandoff: BoundedHandoff; hiddenLabelsExposed: boolean; masteryClaim: boolean };
export type LearnerSessionInput = { caseId: ID; learningGoal: string; sourceEvidence: Evidence[]; authorizedPolicyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; changedEvidenceIds: ID[]; learnerAudience: string; observedResponse: ResponseObservation; attemptLimit?: number };
