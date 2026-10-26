export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: ProjectLearningGoal } };
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
export type LearningProject = RecordMeta & {
  learnerGroupIds: ID[];
  competencyIds: ID[];
  projectBrief: string;
  instructorId: ID;
  timeBudgetHours: number;
};
export type ProjectLearningGoal = RecordMeta & {
  caseId: ID;
  competencyId: ID;
  observableEvidence: string;
  prerequisiteIds: ID[];
  assessmentBoundary: string;
};
export type ProjectConcept = RecordMeta & {
  caseId: ID;
  setting: string;
  roles: string[];
  centralChallenge: string;
  deliverables: string[];
};
export type ProjectMilestone = RecordMeta & {
  caseId: ID;
  deliverable: string;
  prerequisiteIds: ID[];
  evidenceCriteria: string[];
  dueAt: DateTime;
};
export type ScaffoldingPlan = RecordMeta & {
  caseId: ID;
  learnerGroupId: ID;
  observedGap: string;
  support: string;
  fadingCondition: string;
};
export type ProjectRevision = RecordMeta & {
  caseId: ID;
  milestoneId: ID;
  complication: string;
  recoveryOptions: string[];
  preservedGoals: string[];
};
export type DraftProjectLearningGoal = { values: { caseId?: ID; competencyId?: ID; observableEvidence?: string; prerequisiteIds?: ID[]; assessmentBoundary?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProjectLearningGoal = { accepted: ProjectLearningGoal[]; unresolved: DraftProjectLearningGoal[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftProjectConcept = { values: { caseId?: ID; setting?: string; roles?: string[]; centralChallenge?: string; deliverables?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProjectConcept = { accepted: ProjectConcept[]; unresolved: DraftProjectConcept[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftProjectMilestone = { values: { caseId?: ID; deliverable?: string; prerequisiteIds?: ID[]; evidenceCriteria?: string[]; dueAt?: DateTime }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProjectMilestone = { accepted: ProjectMilestone[]; unresolved: DraftProjectMilestone[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftScaffoldingPlan = { values: { caseId?: ID; learnerGroupId?: ID; observedGap?: string; support?: string; fadingCondition?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultScaffoldingPlan = { accepted: ScaffoldingPlan[]; unresolved: DraftScaffoldingPlan[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftProjectRevision = { values: { caseId?: ID; milestoneId?: ID; complication?: string; recoveryOptions?: string[]; preservedGoals?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProjectRevision = { accepted: ProjectRevision[]; unresolved: DraftProjectRevision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; translateGoals: StageResultProjectLearningGoal; designProject: StageResultProjectConcept; sequenceMilestones: StageResultProjectMilestone; adaptScaffolding: StageResultScaffoldingPlan; introduceComplications: StageResultProjectRevision; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { translateGoals: StageResultProjectLearningGoal; designProject: StageResultProjectConcept; sequenceMilestones: StageResultProjectMilestone; adaptScaffolding: StageResultScaffoldingPlan; introduceComplications: StageResultProjectRevision; };
