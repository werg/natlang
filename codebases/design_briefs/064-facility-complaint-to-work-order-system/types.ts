export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type FacilityServiceCase = { caseId: ID; initialObservation: FacilityObservation; locationAliases: FacilityLocationAlias[]; officialLocations: FacilityLocation[]; timePatterns: ComplaintTimePattern[]; accessPermissions: VisitAccessPermission[]; repairHistory: FacilityRepairHistory[]; relatedReports: OccupantReport[]; updatePreferences: PrivacyUpdatePreference[]; evidence: Evidence[] };
export type FacilityLocationAlias = { alias: string; officialLocationId: ID; verifiedBy: Ref | null };
export type FacilityLocation = { locationId: ID; officialName: string; assetIds: ID[] };
export type ComplaintTimePattern = { issueId: ID; observedWindows: string[]; conditions: string[]; evidenceRefs: Ref[] };
export type VisitAccessPermission = { locationId: ID; permittedWindows: string[]; contactMethod: string; consentRef: Ref | null; expiresAt: string | null };
export type FacilityRepairHistory = { issueId: ID; locationId: ID; mechanism: string; priorWorkOrder: ID; verifiedResult: string; recurrenceEvidence: Ref[] };
export type OccupantReport = { reportId: ID; locationId: ID; issueDescription: string; reporterId: ID; shareScope: string; evidenceRef: Ref };
export type PrivacyUpdatePreference = { recipientId: ID; permittedFields: string[]; caseAlias: string; consentRef: Ref };
export type FacilityDispatchPacket = { caseId: ID; officialLocationId: ID | null; locationBlockers: string[]; linkedIssueIds: ID[]; workOrderTasks: string[]; visitWindow: string[]; accessBlockers: string[]; observationPlan: string[]; privateUpdatePackets: string[]; mechanismEscalation: string[]; workBlockers: string[]; dispatchAllowed: boolean; commitAllowed: boolean };
export type FacilityLocationResolution = { locationId: ID | null; alias: string; blockers: string[]; candidates: ID[] };

export type FacilityIssueCluster = { issueIds: ID[]; sharedMechanism: string; reporterPartitions: string[]; unresolvedLinks: string[] };
export type ConsentedVisitPlan = { windows: string[]; blockers: string[]; failedAttemptRecord: string };
export type FacilityWorkOrderPlan = { tasks: string[]; observationPlan: string[]; escalation: string[]; blockers: string[] };
export type FacilityVisitReceipt = { receiptId: ID; caseId: ID; locationId: ID; issueIds: ID[]; accessWindow: string; outcome: "completed" | "no-access" | "cancelled" | "partial" | "unknown"; occurredAt: DateTime; consentRef: Ref | null; completedWorkOrderIds: ID[]; completedCheckIds: ID[]; technicianEvidence: Ref[]; occupantResponse: Ref | null; residualSymptom: string };
export type FacilityVisitLedger = { revision: number; attempts: FacilityVisitReceipt[]; consumedReceiptIds: ID[] };
export type FacilityVisitReconciliation = { ledger: FacilityVisitLedger; outcome: "recorded" | "replayed" | "held"; resolution: "resolved" | "monitor-recurrence" | "access-retry-needed" | "consent-needed" | "unknown"; nextActions: string[] };
export type FacilityVisitAssessment = { binding: "valid" | "mismatch" | "unknown"; resolution: "resolved" | "monitor-recurrence" | "access-retry-needed" | "consent-needed" | "unknown"; missingCheckIds: ID[]; reasons: string[]; nextActions: string[] };

export type FacilityRecurrenceFinding = { issueId: ID; relationship: "recurrence" | "compatible" | "different" | "unknown"; mechanism: string; priorWorkOrderId: ID; evidenceRefs: Ref[]; blockers: string[] };
export type FacilityReportLink = { reportId: ID; compatibleIssueIds: ID[]; relation: "same-mechanism" | "separate" | "unresolved"; blockers: string[]; shareScope: string };
export type FacilityObservationGate = { observationPlan: string[]; blockers: string[] };
