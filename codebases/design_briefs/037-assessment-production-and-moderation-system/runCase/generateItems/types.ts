export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: AssessmentBlueprint } };
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
export type AssessmentProject = RecordMeta & {
  competencyIds: ID[];
  audience: string;
  durationMinutes: number;
  assessmentOwnerId: ID;
  stakes: string;
};
export type AssessmentBlueprint = RecordMeta & {
  caseId: ID;
  competencyIds: ID[];
  taskKinds: string[];
  coverageWeights: Ratio[];
  excludedConstructs: string[];
};
export type AssessmentItem = RecordMeta & {
  caseId: ID;
  competencyIds: ID[];
  prompt: string;
  responseFormat: string;
  allowedResources: string[];
};
export type ScoringRubric = RecordMeta & {
  caseId: ID;
  itemId: ID;
  criteria: string[];
  scoreLevels: string[];
  validAlternativeAnswers: string[];
};
export type ItemChallenge = RecordMeta & {
  caseId: ID;
  itemId: ID;
  ambiguousInterpretation: string;
  counterexampleAnswer: string;
  proposedRepair: string;
};
export type ModerationDecision = RecordMeta & {
  caseId: ID;
  responseId: ID;
  originalScore: number;
  reviewedScore: number;
  evidenceReason: string;
};
export type DraftAssessmentBlueprint = { values: { caseId?: ID; competencyIds?: ID[]; taskKinds?: string[]; coverageWeights?: Ratio[]; excludedConstructs?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultAssessmentBlueprint = { accepted: AssessmentBlueprint[]; unresolved: DraftAssessmentBlueprint[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftAssessmentItem = { values: { caseId?: ID; competencyIds?: ID[]; prompt?: string; responseFormat?: string; allowedResources?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultAssessmentItem = { accepted: AssessmentItem[]; unresolved: DraftAssessmentItem[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftScoringRubric = { values: { caseId?: ID; itemId?: ID; criteria?: string[]; scoreLevels?: string[]; validAlternativeAnswers?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultScoringRubric = { accepted: ScoringRubric[]; unresolved: DraftScoringRubric[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftItemChallenge = { values: { caseId?: ID; itemId?: ID; ambiguousInterpretation?: string; counterexampleAnswer?: string; proposedRepair?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultItemChallenge = { accepted: ItemChallenge[]; unresolved: DraftItemChallenge[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftModerationDecision = { values: { caseId?: ID; responseId?: ID; originalScore?: number; reviewedScore?: number; evidenceReason?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultModerationDecision = { accepted: ModerationDecision[]; unresolved: DraftModerationDecision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; blueprintCoverage: StageResultAssessmentBlueprint; generateItems: StageResultAssessmentItem; constructRubrics: StageResultScoringRubric; challengeAssessment: StageResultItemChallenge; moderateAndAppeal: StageResultModerationDecision; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { blueprintCoverage: StageResultAssessmentBlueprint; generateItems: StageResultAssessmentItem; constructRubrics: StageResultScoringRubric; challengeAssessment: StageResultItemChallenge; moderateAndAppeal: StageResultModerationDecision; };
