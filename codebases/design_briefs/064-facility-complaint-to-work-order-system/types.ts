export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: FacilityObservation } };
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
export type FacilityCase = RecordMeta & {
  buildingId: ID;
  occupantId: ID;
  reportedProblem: string;
  accessConstraints: string[];
  coordinatorId: ID;
};
export type FacilityObservation = RecordMeta & {
  caseId: ID;
  locationId: ID;
  assetCandidateIds: ID[];
  symptom: string;
  occurrencePattern: string;
};
export type FacilityIssueGroup = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  sharedMechanism: string;
  separateIssueIds: ID[];
  uncertainty: string;
};
export type FacilityWorkOrder = RecordMeta & {
  caseId: ID;
  issueIds: ID[];
  trade: string;
  task: string;
  completionEvidence: string;
};
export type FacilityVisit = RecordMeta & {
  caseId: ID;
  workOrderIds: ID[];
  assignedPartyId: ID;
  accessWindow: string;
  prerequisiteIds: ID[];
};
export type FacilityResolution = RecordMeta & {
  caseId: ID;
  workOrderId: ID;
  technicianEvidence: Ref[];
  occupantResponse: Ref;
  residualSymptom: string;
};
export type DraftFacilityObservation = { values: { caseId?: ID; locationId?: ID; assetCandidateIds?: ID[]; symptom?: string; occurrencePattern?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFacilityObservation = { accepted: FacilityObservation[]; unresolved: DraftFacilityObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFacilityIssueGroup = { values: { caseId?: ID; observationIds?: ID[]; sharedMechanism?: string; separateIssueIds?: ID[]; uncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFacilityIssueGroup = { accepted: FacilityIssueGroup[]; unresolved: DraftFacilityIssueGroup[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFacilityWorkOrder = { values: { caseId?: ID; issueIds?: ID[]; trade?: string; task?: string; completionEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFacilityWorkOrder = { accepted: FacilityWorkOrder[]; unresolved: DraftFacilityWorkOrder[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFacilityVisit = { values: { caseId?: ID; workOrderIds?: ID[]; assignedPartyId?: ID; accessWindow?: string; prerequisiteIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFacilityVisit = { accepted: FacilityVisit[]; unresolved: DraftFacilityVisit[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFacilityResolution = { values: { caseId?: ID; workOrderId?: ID; technicianEvidence?: Ref[]; occupantResponse?: Ref; residualSymptom?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFacilityResolution = { accepted: FacilityResolution[]; unresolved: DraftFacilityResolution[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretComplaint: StageResultFacilityObservation; linkRelatedIssues: StageResultFacilityIssueGroup; formWorkOrders: StageResultFacilityWorkOrder; coordinateVisits: StageResultFacilityVisit; verifyResolution: StageResultFacilityResolution; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretComplaint: StageResultFacilityObservation; linkRelatedIssues: StageResultFacilityIssueGroup; formWorkOrders: StageResultFacilityWorkOrder; coordinateVisits: StageResultFacilityVisit; verifyResolution: StageResultFacilityResolution; };

export type FacilityServiceCase = { caseId: ID; initialObservation: FacilityObservation; locationAliases: FacilityLocationAlias[]; officialLocations: FacilityLocation[]; timePatterns: ComplaintTimePattern[]; accessPermissions: VisitAccessPermission[]; repairHistory: FacilityRepairHistory[]; relatedReports: OccupantReport[]; updatePreferences: PrivacyUpdatePreference[]; evidence: Evidence[] };
export type FacilityLocationAlias = { alias: string; officialLocationId: ID; verifiedBy: Ref | null };
export type FacilityLocation = { locationId: ID; officialName: string; assetIds: ID[] };
export type ComplaintTimePattern = { issueId: ID; observedWindows: string[]; conditions: string[]; evidenceRefs: Ref[] };
export type VisitAccessPermission = { locationId: ID; permittedWindows: string[]; contactMethod: string; consentRef: Ref | null; expiresAt: string | null };
export type FacilityRepairHistory = { issueId: ID; locationId: ID; mechanism: string; priorWorkOrder: ID; verifiedResult: string; recurrenceEvidence: Ref[] };
export type OccupantReport = { reportId: ID; locationId: ID; issueDescription: string; reporterId: ID; shareScope: string; evidenceRef: Ref };
export type PrivacyUpdatePreference = { recipientId: ID; permittedFields: string[]; caseAlias: string; consentRef: Ref };
export type FacilityDispatchPacket = { caseId: ID; officialLocationId: ID | null; locationBlockers: string[]; linkedIssueIds: ID[]; workOrderTasks: string[]; visitWindow: string[]; accessBlockers: string[]; observationPlan: string[]; privateUpdatePackets: string[]; mechanismEscalation: string[]; dispatchAllowed: boolean; commitAllowed: boolean };
export type FacilityLocationResolution = { locationId: ID | null; alias: string; blockers: string[]; candidates: ID[] };

export type FacilityIssueCluster = { issueIds: ID[]; sharedMechanism: string; reporterPartitions: string[]; unresolvedLinks: string[] };
export type ConsentedVisitPlan = { windows: string[]; blockers: string[]; failedAttemptRecord: string };
export type FacilityWorkOrderPlan = { tasks: string[]; observationPlan: string[]; escalation: string[] };
