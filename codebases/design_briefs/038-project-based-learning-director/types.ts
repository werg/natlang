export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type ProjectDirectorRequest = { projectId: ID; seedGoal: ProjectLearningGoal; learningIntent: string; sourceEvidence: Evidence[]; authorizedConstraints: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; changedEvidenceIds: ID[]; learnerGroup: string; resourceInventory: string[]; availableWindow: string; requiredGoalIds: ID[]; prior?: ProjectDirectorCheckpoint };
export type LearningEvidenceMap = { goals: { goalId: ID; demonstration: string; prerequisiteSkills: string[]; resourceNeeds: string[]; feasible: boolean; evidenceOwner: string }[]; unsupportedGoalIds: ID[] };
export type CollaborativeProjectConcept = { challenge: string; roles: { learner: string; decision: string; attributableArtifact: string; goalIds: ID[] }[]; materialRoutes: { missingMaterial: string; alternatives: string[]; preservedGoalIds: ID[]; unresolved: string[] }[] };
export type Milestone = { id: ID; deliverable: string; prerequisites: ID[]; learnerOwners: string[]; dueWindow: string; resourceNeeds: string[]; goalEvidenceIds: ID[] };
export type MilestoneOrder = { orderedIds: ID[]; blockedIds: ID[]; cycleOrMissingPrerequisites: string[] };
export type MilestoneCalendar = { milestones: Milestone[]; dependencyOrder?: MilestoneOrder; overloadedLearners: string[]; missedMilestoneRecovery: string[] };
export type ScaffoldPlan = { learnerSupports: { learner: string; milestoneId: ID; support: string; learnerMustProduce: string[] }[]; individualDemonstrations: string[]; authorshipQuestions: string[]; completionEvidenceGeneratedBySystem: string[] };
export type ComplicationDecision = { remainingFeasibleGoals: ID[]; blockedGoals: ID[]; alternateRoutes: string[]; releaseBlockers: string[]; creativeMethodReviews: string[] };
export type ProjectDirectorCheckpoint = { projectId: ID; version: number; completedMilestoneIds: ID[]; individualEvidenceDue: string[]; scaffoldExposure: string[]; unresolvedQuestions: string[] };
export type ProjectReadiness = { status: 'blocked' | 'review' | 'proposed'; reasons: string[] };
export type ProjectDirectorProposal = { projectId: ID; goals: LearningEvidenceMap; concept: CollaborativeProjectConcept; calendar: MilestoneCalendar; scaffolds: ScaffoldPlan; complication: ComplicationDecision; checkpoint: ProjectDirectorCheckpoint; readiness: ProjectReadiness; commitAllowed: boolean };
export type RequiredLearningGoal = { id: ID; description: string; excludedConstructs: string[] };
export type GoalFacet = { goalId: ID; behavior: string; possibleArtifact: string; exclusions: string[] };
export type FacetEvidenceMatch = { goalId: ID; artifact: string; learnerOwner: string; resourceDependencies: string[]; missingConditions: string[] };
export type RoleContribution = { learner: string; consequentialDecision: string; attributableArtifact: string; goalIds: ID[] };
export type MaterialEquivalenceReview = { unavailable: string; alternative: string; preservedGoalIds: ID[]; unresolvedSafetyOrAccess: string[]; status: 'equivalent' | 'review' | 'not-equivalent' };
export type LearnerLoadAssessment = { learner: string; totalPlannedWork: string; availableWindow: string; overloaded: boolean; bottleneckMilestoneIds: ID[] }[];
export type MilestoneRecoveryPlan = { blockedMilestoneIds: ID[]; redistributedLogistics: string[]; learnerEvidenceStillDue: string[]; capacityConflicts: string[] };
export type AuthorshipRiskReview = { scaffoldIdsCompletingAssessedWork: ID[]; sharedArtifactsWithUnclearOwner: ID[]; individualEvidenceDue: string[] };
export type IndividualDemonstrationRequest = { learner: string; goalId: ID; prompt: string; acceptableObservation: string };
export type DeadlineFeasibility = { remainingGoalIds: ID[]; infeasibleGoalIds: ID[]; overloadedLearners: string[]; unresolvedResources: string[] };
export type CreativeRouteAssessment = { goalId: ID; method: string; competencyEvidence: string; status: 'demonstrates' | 'insufficient' | 'review' };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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
