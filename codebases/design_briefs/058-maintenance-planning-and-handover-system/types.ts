export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: AssetObservation } };
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
export type MaintenanceCase = RecordMeta & {
  assetId: ID;
  observedIssue: string;
  maintenanceOwnerId: ID;
  serviceWindow: string;
  procedureIds: ID[];
};
export type AssetObservation = RecordMeta & {
  caseId: ID;
  assetId: ID;
  symptom: string;
  operatingContext: string;
  priorWorkRefs: Ref[];
};
export type MaintenanceProcedureFit = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  procedureId: ID;
  applicability: string;
  missingPrerequisite: string;
};
export type MaintenanceWorkOrder = RecordMeta & {
  caseId: ID;
  procedureId: ID;
  taskIds: ID[];
  partIds: ID[];
  prerequisiteIds: ID[];
};
export type MaintenanceProgress = RecordMeta & {
  caseId: ID;
  taskId: ID;
  completedAction: string;
  verificationRef: Ref;
  remainingState: string;
};
export type MaintenanceHandover = RecordMeta & {
  caseId: ID;
  openTaskIds: ID[];
  assetState: string;
  nextRequiredChecks: string[];
  receivingOwnerId: ID;
};
export type DraftAssetObservation = { values: { caseId?: ID; assetId?: ID; symptom?: string; operatingContext?: string; priorWorkRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultAssetObservation = { accepted: AssetObservation[]; unresolved: DraftAssetObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMaintenanceProcedureFit = { values: { caseId?: ID; observationIds?: ID[]; procedureId?: ID; applicability?: string; missingPrerequisite?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMaintenanceProcedureFit = { accepted: MaintenanceProcedureFit[]; unresolved: DraftMaintenanceProcedureFit[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMaintenanceWorkOrder = { values: { caseId?: ID; procedureId?: ID; taskIds?: ID[]; partIds?: ID[]; prerequisiteIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMaintenanceWorkOrder = { accepted: MaintenanceWorkOrder[]; unresolved: DraftMaintenanceWorkOrder[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMaintenanceProgress = { values: { caseId?: ID; taskId?: ID; completedAction?: string; verificationRef?: Ref; remainingState?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMaintenanceProgress = { accepted: MaintenanceProgress[]; unresolved: DraftMaintenanceProgress[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftMaintenanceHandover = { values: { caseId?: ID; openTaskIds?: ID[]; assetState?: string; nextRequiredChecks?: string[]; receivingOwnerId?: ID }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultMaintenanceHandover = { accepted: MaintenanceHandover[]; unresolved: DraftMaintenanceHandover[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretObservations: StageResultAssetObservation; selectProcedure: StageResultMaintenanceProcedureFit; constructWorkPlan: StageResultMaintenanceWorkOrder; interpretProgress: StageResultMaintenanceProgress; composeHandover: StageResultMaintenanceHandover; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretObservations: StageResultAssetObservation; selectProcedure: StageResultMaintenanceProcedureFit; constructWorkPlan: StageResultMaintenanceWorkOrder; interpretProgress: StageResultMaintenanceProgress; composeHandover: StageResultMaintenanceHandover; };

export type MaintenanceShiftCase = { caseId: ID; priorObservation: AssetObservation; assetIdentity: AssetIdentityEvidence; reportedObservations: MaintenanceObservation[]; plannedTasks: MaintenanceTask[]; completionReports: CompletionReport[]; partChanges: PartRevisionChange[]; isolation: IsolationCheckpoint | null; faultHistory: FaultRecurrence[]; incomingShift: string; outgoingShift: string; procedures: MaintenanceProcedureFit[]; evidence: Evidence[] };
export type AssetIdentityEvidence = { officialIds: ID[]; aliases: string[]; locationRefs: Ref[]; verifiedRef: Ref | null };
export type MaintenanceObservation = { assetId: ID; symptom: string; condition: string; observedAt: string; evidenceRef: Ref };
export type MaintenanceTask = { taskId: ID; assetId: ID; procedureId: ID; prerequisiteIds: ID[]; requiredChecks: string[]; status: string };
export type CompletionReport = { taskId: ID; reportedDone: boolean; verificationRefs: Ref[]; verifiedChecks: string[]; reportedAt: string };
export type PartRevisionChange = { taskId: ID; specifiedPart: ID; offeredPart: ID; specifiedRevision: string; offeredRevision: string; compatibilityEvidence: Ref[] };
export type IsolationCheckpoint = { assetId: ID; isolationState: string; lockRefs: Ref[]; handoverChecks: string[]; verificationOwner: string };
export type FaultRecurrence = { assetId: ID; symptom: string; priorDisappearance: string; monitoringRule: string; evidenceRefs: Ref[] };
export type ShiftHandoverProposal = { officialAssetId: ID | null; identityBlockers: string[]; orderedTasks: ID[]; incompleteTaskIds: ID[]; rejectedPartChanges: string[]; isolationState: IsolationCheckpoint | null; monitoringCarryover: string[]; handoverNote: string; dispatchAllowed: boolean };
export type AssetResolution = { officialAssetId: ID | null; candidates: ID[]; blockers: string[]; evidenceRefs: Ref[] };
