export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: ReuseSiteFact } };
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
export type ReuseStudy = RecordMeta & {
  buildingId: ID;
  stakeholderIds: ID[];
  suppliedConstraintIds: ID[];
  studyOwnerId: ID;
  budgetRange: string;
};
export type ReuseSiteFact = RecordMeta & {
  caseId: ID;
  buildingArea: string;
  observedCondition: string;
  sourceRef: Ref;
  certainty: string;
};
export type ReuseGoal = RecordMeta & {
  caseId: ID;
  stakeholderId: ID;
  desiredFunction: string;
  essentialConstraint: string;
  negotiablePreference: string;
};
export type ReuseConcept = RecordMeta & {
  caseId: ID;
  goalIds: ID[];
  spatialApproach: string;
  phasedUse: string;
  unresolvedAssumptions: string[];
};
export type ReuseInvestigation = RecordMeta & {
  caseId: ID;
  conceptIds: ID[];
  unknownCondition: string;
  proposedSurvey: string;
  decisionConsequence: string;
};
export type ReuseOptionReview = RecordMeta & {
  caseId: ID;
  conceptId: ID;
  costBasis: string;
  constraintFindings: string[];
  residualRisk: string;
};
export type DraftReuseSiteFact = { values: { caseId?: ID; buildingArea?: string; observedCondition?: string; sourceRef?: Ref; certainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseSiteFact = { accepted: ReuseSiteFact[]; unresolved: DraftReuseSiteFact[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseGoal = { values: { caseId?: ID; stakeholderId?: ID; desiredFunction?: string; essentialConstraint?: string; negotiablePreference?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseGoal = { accepted: ReuseGoal[]; unresolved: DraftReuseGoal[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseConcept = { values: { caseId?: ID; goalIds?: ID[]; spatialApproach?: string; phasedUse?: string; unresolvedAssumptions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseConcept = { accepted: ReuseConcept[]; unresolved: DraftReuseConcept[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseInvestigation = { values: { caseId?: ID; conceptIds?: ID[]; unknownCondition?: string; proposedSurvey?: string; decisionConsequence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseInvestigation = { accepted: ReuseInvestigation[]; unresolved: DraftReuseInvestigation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseOptionReview = { values: { caseId?: ID; conceptId?: ID; costBasis?: string; constraintFindings?: string[]; residualRisk?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseOptionReview = { accepted: ReuseOptionReview[]; unresolved: DraftReuseOptionReview[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; recoverSiteFacts: StageResultReuseSiteFact; interpretGoals: StageResultReuseGoal; generateConcepts: StageResultReuseConcept; planInvestigations: StageResultReuseInvestigation; compareFeasibility: StageResultReuseOptionReview; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { recoverSiteFacts: StageResultReuseSiteFact; interpretGoals: StageResultReuseGoal; generateConcepts: StageResultReuseConcept; planInvestigations: StageResultReuseInvestigation; compareFeasibility: StageResultReuseOptionReview; };

export type ReuseFeasibilityDossier = { caseId: ID; siteFacts: ReuseSiteFact[]; historicDrawings: HistoricPlan[]; surveys: CurrentSurvey[]; intendedUses: AdaptiveReuseGoal[]; constraints: ReuseConstraint[]; concepts: ReuseConceptOption[]; structuralAssumptions: StructuralAssumption[]; investigations: InvestigationProposal[]; costScopes: CostScope[]; phases: ReusePhase[]; circulationRoutes: TemporaryRoute[]; evidence: Evidence[]; state?: ReuseFeasibilityState };
export type HistoricPlan = { drawingId: ID; revision: string; areaId: ID; feature: string; sourceDate: string; evidenceRef: Ref };
export type CurrentSurvey = { surveyId: ID; areaId: ID; feature: string; measuredValue: string; method: string; qualified: boolean; observedAt: string; evidenceRef: Ref };
export type AdaptiveReuseGoal = { goalId: ID; intendedUse: string; capacity: string; operatingHours: string; changedAt: string; authorityRef: Ref };
export type ReuseConstraint = { constraintId: ID; areaId: ID; kind: string; limit: string; evidenceRef: Ref };
export type ReuseConceptOption = { optionId: ID; affectedAreas: ID[]; enablingWorks: ID[]; assumedCapacity: string; requiredInvestigations: ID[]; phaseIds: ID[] };
export type StructuralAssumption = { assumptionId: ID; areaId: ID; claim: string; dependsOn: string[]; qualifiedReviewRequired: boolean; evidenceRef: Ref | null };
export type InvestigationProposal = { investigationId: ID; areaId: ID; question: string; qualification: string; dependencyOptionIds: ID[]; status: string };
export type CostScope = { costId: ID; optionId: ID; workType: string; included: boolean; amountBasis: string; evidenceRef: Ref };
export type ReusePhase = { phaseId: ID; areaIds: ID[]; openToPublic: boolean; activeConstructionAreas: ID[]; temporaryRoutes: ID[] };
export type TemporaryRoute = { routeId: ID; fromArea: ID; toArea: ID; accessible: boolean; conflictsWith: ID[]; evidenceRef: Ref };
export type SiteEvidenceReconciliation = { preferredCurrentFacts: ID[]; historicalConflicts: string[]; unverifiedAreas: ID[] };
export type ReuseFeasibilityReview = { caseId: ID; evidence: SiteEvidenceReconciliation; affectedConstraints: ID[]; contingentOptions: ID[]; investigationGates: string[]; missingEnablingWorks: ID[]; phaseAccessConflicts: ID[]; optionDisposition: string; state: ReuseFeasibilityState; commitAllowed: boolean };
export type ConstraintReassessment = { affectedIds: ID[]; blockers: string[]; missingEvidence: string[] };
export type StructuralReviewGate = { contingentOptionIds: ID[]; reviewTasks: string[] };
export type ReuseFeasibilityRevision = { sourceKey: string; revision: number; caseId: ID; siteFacts: ReuseSiteFact[]; historicDrawings: HistoricPlan[]; surveys: CurrentSurvey[]; intendedUses: AdaptiveReuseGoal[]; constraints: ReuseConstraint[]; concepts: ReuseConceptOption[]; structuralAssumptions: StructuralAssumption[]; investigations: InvestigationProposal[]; costScopes: CostScope[]; phases: ReusePhase[]; circulationRoutes: TemporaryRoute[]; evidence: Evidence[]; evidenceReview: SiteEvidenceReconciliation; affectedConstraints: ID[]; contingentOptions: ID[]; investigationGates: string[]; missingEnablingWorks: ID[]; phaseAccessConflicts: ID[]; optionDisposition: string };
export type ReuseFeasibilityState = { revision: number; currentSourceKey: string | null; revisions: ReuseFeasibilityRevision[]; supersededSurveyIds: ID[]; invalidatedInvestigationIds: ID[] };
