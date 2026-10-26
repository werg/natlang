export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: PracticeScenario } };
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
export type PracticeSession = RecordMeta & {
  learnerId: ID;
  scenarioFamily: string;
  procedureId: ID;
  instructorId: ID;
  difficulty: string;
};
export type PracticeScenario = RecordMeta & {
  caseId: ID;
  initialFacts: string[];
  hiddenFacts: string[];
  actorGoals: string[];
  allowedActions: string[];
};
export type LearnerAction = RecordMeta & {
  caseId: ID;
  utterance: string;
  intendedAction: string;
  targetId: ID;
  ambiguity: string;
};
export type PracticeTransition = RecordMeta & {
  caseId: ID;
  actionId: ID;
  stateDelta: string;
  revealedFacts: string[];
  consequence: string;
};
export type PracticeComplication = RecordMeta & {
  caseId: ID;
  trigger: string;
  newFact: string;
  affectedObligationIds: ID[];
  recoveryPaths: string[];
};
export type PracticeFeedback = RecordMeta & {
  caseId: ID;
  actionIds: ID[];
  demonstratedSkills: string[];
  missedEvidence: string[];
  nextPractice: string;
};
export type DraftPracticeScenario = { values: { caseId?: ID; initialFacts?: string[]; hiddenFacts?: string[]; actorGoals?: string[]; allowedActions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultPracticeScenario = { accepted: PracticeScenario[]; unresolved: DraftPracticeScenario[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftLearnerAction = { values: { caseId?: ID; utterance?: string; intendedAction?: string; targetId?: ID; ambiguity?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultLearnerAction = { accepted: LearnerAction[]; unresolved: DraftLearnerAction[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftPracticeTransition = { values: { caseId?: ID; actionId?: ID; stateDelta?: string; revealedFacts?: string[]; consequence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultPracticeTransition = { accepted: PracticeTransition[]; unresolved: DraftPracticeTransition[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftPracticeComplication = { values: { caseId?: ID; trigger?: string; newFact?: string; affectedObligationIds?: ID[]; recoveryPaths?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultPracticeComplication = { accepted: PracticeComplication[]; unresolved: DraftPracticeComplication[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftPracticeFeedback = { values: { caseId?: ID; actionIds?: ID[]; demonstratedSkills?: string[]; missedEvidence?: string[]; nextPractice?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultPracticeFeedback = { accepted: PracticeFeedback[]; unresolved: DraftPracticeFeedback[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; authorSituation: StageResultPracticeScenario; interpretAction: StageResultLearnerAction; advanceWorld: StageResultPracticeTransition; introduceChallenge: StageResultPracticeComplication; debriefPerformance: StageResultPracticeFeedback; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { authorSituation: StageResultPracticeScenario; interpretAction: StageResultLearnerAction; advanceWorld: StageResultPracticeTransition; introduceChallenge: StageResultPracticeComplication; debriefPerformance: StageResultPracticeFeedback; };
