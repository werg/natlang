export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: HabitatObservation } };
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
export type RestorationProject = RecordMeta & {
  siteId: ID;
  landholderIds: ID[];
  restorationGoals: string[];
  expertOwnerId: ID;
  planningHorizon: string;
};
export type HabitatObservation = RecordMeta & {
  caseId: ID;
  locationId: ID;
  observedFeature: string;
  surveyMethod: string;
  seasonalContext: string;
};
export type RestorationObjective = RecordMeta & {
  caseId: ID;
  desiredCondition: string;
  indicator: string;
  landholderConstraint: string;
  uncertainty: string;
};
export type RestorationOption = RecordMeta & {
  caseId: ID;
  objectiveIds: ID[];
  approvedAction: string;
  prerequisites: string[];
  expectedMechanism: string;
};
export type MonitoringTask = RecordMeta & {
  caseId: ID;
  objectiveId: ID;
  observationMethod: string;
  samplingWindow: string;
  decisionRule: string;
};
export type RestorationRevision = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  affectedOptionIds: ID[];
  proposedChange: string;
  expertApprovalRef: Ref;
};
export type DraftHabitatObservation = { values: { caseId?: ID; locationId?: ID; observedFeature?: string; surveyMethod?: string; seasonalContext?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHabitatObservation = { accepted: HabitatObservation[]; unresolved: DraftHabitatObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftRestorationObjective = { values: { caseId?: ID; desiredCondition?: string; indicator?: string; landholderConstraint?: string; uncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultRestorationObjective = { accepted: RestorationObjective[]; unresolved: DraftRestorationObjective[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftRestorationOption = { values: { caseId?: ID; objectiveIds?: ID[]; approvedAction?: string; prerequisites?: string[]; expectedMechanism?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultRestorationOption = { accepted: RestorationOption[]; unresolved: DraftRestorationOption[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMonitoringTask = { values: { caseId?: ID; objectiveId?: ID; observationMethod?: string; samplingWindow?: string; decisionRule?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMonitoringTask = { accepted: MonitoringTask[]; unresolved: DraftMonitoringTask[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftRestorationRevision = { values: { caseId?: ID; observationIds?: ID[]; affectedOptionIds?: ID[]; proposedChange?: string; expertApprovalRef?: Ref }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultRestorationRevision = { accepted: RestorationRevision[]; unresolved: DraftRestorationRevision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretBaseline: StageResultHabitatObservation; translateObjectives: StageResultRestorationObjective; constructInterventions: StageResultRestorationOption; planMonitoring: StageResultMonitoringTask; adaptManagement: StageResultRestorationRevision; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretBaseline: StageResultHabitatObservation; translateObjectives: StageResultRestorationObjective; constructInterventions: StageResultRestorationOption; planMonitoring: StageResultMonitoringTask; adaptManagement: StageResultRestorationRevision; };

export type RestorationPlanningInput = { caseId: ID; baseline: HabitatObservation[]; objectives: RestorationObjectiveRecord[]; surveys: HabitatSurvey[]; speciesLocations: SensitiveSpeciesRecord[]; interventions: RestorationIntervention[]; landPermissions: FieldAccessPermission[]; impactReports: InterventionImpact[]; volunteerSamples: VolunteerSample[]; expertRoster: ExpertReviewScope[]; evidence: Evidence[] };
export type RestorationObjectiveRecord = { objectiveId: ID; target: string; areaIds: ID[]; indicatorIds: ID[]; seasonWindow: string; authorityRef: Ref };
export type HabitatSurvey = { surveyId: ID; areaId: ID; method: string; season: string; observedAt: string; effort: string; observations: string[]; evidenceRef: Ref };
export type SensitiveSpeciesRecord = { recordId: ID; exactCoordinate: string; sensitivity: string; accessRule: string; evidenceRef: Ref };
export type RestorationIntervention = { interventionId: ID; areaIds: ID[]; action: string; objectiveIds: ID[]; permissionIds: ID[]; stopConditions: string[]; qualifiedLead: string };
export type FieldAccessPermission = { permissionId: ID; areaId: ID; permittedActions: string[]; expiresAt: string; ownerRef: Ref };
export type InterventionImpact = { reportId: ID; interventionId: ID; observation: string; severity: string; observedAt: string; evidenceRef: Ref };
export type VolunteerSample = { sampleId: ID; areaId: ID; method: string; season: string; effortDuration: string; observerCount: number; observations: string[]; evidenceRef: Ref };
export type ExpertReviewScope = { expertId: ID; taxaOrSystem: string[]; qualificationRef: Ref; available: boolean };
export type SpeciesDisclosurePlan = { volunteerView: string[]; publicView: string[]; exactLocationView: string[]; restrictedRecordIds: ID[] };
export type SurveyComparison = { comparablePairs: string[]; incomparablePairs: string[]; confoundingChanges: string[] };
export type MonitoringDesign = { objectiveIndicators: string[]; seasonalRounds: string[]; effortFields: string[]; repeatMethods: string[]; comparabilityWarnings: string[] };
export type HabitatActionPlan = { caseId: ID; disclosure: SpeciesDisclosurePlan; comparison: SurveyComparison; monitoring: MonitoringDesign; permittedActions: ID[]; stoppedActions: ID[]; suspendedInterventions: ID[]; expertEscalations: string[]; unresolvedPermissions: ID[]; commitAllowed: boolean };
export type FieldActionAuthorization = { permittedActions: ID[]; stoppedActions: ID[]; unresolvedPermissionIds: ID[] };
export type EcologicalIncidentReview = { suspendedInterventions: ID[]; reviewRequests: string[]; preservedObservationRefs: Ref[] };
