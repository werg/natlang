export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type MaintenanceShiftCase = { caseId: ID; priorObservation: AssetObservation; assetIdentity: AssetIdentityEvidence; reportedObservations: MaintenanceObservation[]; plannedTasks: MaintenanceTask[]; completionReports: CompletionReport[]; partChanges: PartRevisionChange[]; isolation: IsolationCheckpoint | null; faultHistory: FaultRecurrence[]; incomingShift: string; outgoingShift: string; procedures: MaintenanceProcedureFit[]; evidence: Evidence[] };
export type AssetIdentityEvidence = { officialIds: ID[]; aliases: string[]; locationRefs: Ref[]; verifiedRef: Ref | null };
export type MaintenanceObservation = { assetId: ID; symptom: string; condition: string; observedAt: string; evidenceRef: Ref };
export type MaintenanceTask = { taskId: ID; assetId: ID; procedureId: ID; prerequisiteIds: ID[]; requiredChecks: string[]; status: string };
export type CompletionReport = { taskId: ID; reportedDone: boolean; verificationRefs: Ref[]; verifiedChecks: string[]; reportedAt: string };
export type PartRevisionChange = { taskId: ID; specifiedPart: ID; offeredPart: ID; specifiedRevision: string; offeredRevision: string; compatibilityEvidence: Ref[] };
export type IsolationCheckpoint = { assetId: ID; isolationState: string; lockRefs: Ref[]; handoverChecks: string[]; verificationOwner: string };
export type FaultRecurrence = { assetId: ID; symptom: string; priorDisappearance: string; monitoringRule: string; evidenceRefs: Ref[] };
export type ShiftHandoverProposal = { officialAssetId: ID | null; identityBlockers: string[]; orderedTasks: ID[]; incompleteTaskIds: ID[]; rejectedPartChanges: string[]; isolationState: IsolationCheckpoint | null; monitoringCarryover: string[]; handoverNote: string; dispatchAllowed: boolean };
export type MaintenanceTaskReceipt = { receiptId: ID; caseId: ID; officialAssetId: ID; taskId: ID; outcome: "completed" | "partial" | "not-done" | "unknown"; verifiedChecks: string[]; installedPartId: ID | null; installedRevision: string | null; verificationRefs: Ref[]; isolationStateAtHandover: string; isolationHandoverChecks: string[]; isolationEvidenceRefs: Ref[]; receivedBy: ID; occurredAt: DateTime };
export type MaintenanceExecutionBook = { revision: number; taskReceipts: MaintenanceTaskReceipt[]; completedTaskIds: ID[]; unresolvedTaskIds: ID[]; consumedReceiptIds: ID[] };
export type MaintenanceExecutionReconciliation = { book: MaintenanceExecutionBook; disposition: "recorded" | "replayed" | "held"; remainingTaskIds: ID[]; isolationDisposition: "not-applicable" | "retained" | "handover-verified" | "unknown"; followUp: string[] };
export type MaintenanceReceiptAssessment = { matchesPlannedTask: boolean; verificationComplete: boolean; partCompatible: boolean; reasons: string[] };
export type PlannedTaskSequenceCheck = { eligible: boolean; blockingPrerequisiteIds: ID[]; reasons: string[] };
export type InstalledPartRevisionCheck = { compatible: boolean; reasons: string[]; evidenceRefs: Ref[] };
export type AssetResolution = { officialAssetId: ID | null; candidates: ID[]; blockers: string[]; evidenceRefs: Ref[] };
