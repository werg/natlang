export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type RepairRequest = RecordMeta & {
  requesterId: ID;
  itemDescription: string;
  reportedSymptoms: string[];
  preferredWindow: string;
  coordinatorId: ID;
};
export type RepairSymptom = RecordMeta & {
  caseId: ID;
  itemId: ID;
  observedBehavior: string;
  priorAttempt: string;
  riskFlag: string;
};
export type RepairDiagnostic = RecordMeta & {
  caseId: ID;
  symptomIds: ID[];
  approvedCheck: string;
  requiredSkill: string;
  expectedDistinction: string;
};
export type RepairAssignment = RecordMeta & {
  caseId: ID;
  taskIds: ID[];
  repairerId: ID;
  skillEvidence: Ref[];
  capacitySlot: string;
};
export type RepairWorkPlan = RecordMeta & {
  caseId: ID;
  assignmentIds: ID[];
  partIds: ID[];
  prerequisiteIds: ID[];
  customerApprovalRef: Ref;
};
export type RepairOutcome = RecordMeta & {
  caseId: ID;
  workPlanId: ID;
  verifiedResult: string;
  remainingIssue: string;
  reusableLesson: string;
};
export type DraftRepairSymptom = { values: { caseId?: ID; itemId?: ID; observedBehavior?: string; priorAttempt?: string; riskFlag?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type CooperativeRepairRequest = { requestId: ID; priorSymptom: RepairSymptom; customerId: ID | null; itemId: ID; reportedObservations: string[]; priorAttemptNotes: string[]; riskFlags: string[]; approvedProcedures: ApprovedCheck[]; repairers: Repairer[]; toolWindows: SharedToolWindow[]; parts: SparePart[]; budget: BudgetAuthorization; customerEvidence: Evidence[]; workshopRules: string[]; outcomeChecks: RepairCheck[]; priorState: RepairCheckpoint | null; workReceipts?: RepairWorkReceipt[] };
export type ApprovedCheck = { procedureId: ID; instruction: string; distinguishes: string[]; allowedRisk: string; skill: string; stopCondition: string };
export type Repairer = { id: ID; skills: string[]; qualifications: QualificationProof[]; availability: string[]; consentedTasks: string[]; supervisorId: ID | null };
export type QualificationProof = { skill: string; scope: string; expiresOn: string; evidenceRef: Ref };
export type SharedToolWindow = { toolId: ID; window: string; reservedFor: ID[] };
export type SparePart = { partId: ID; compatibilityClaim: string; compatibilityEvidence: Ref[]; quantity: number; cost: string };
export type BudgetAuthorization = { limit: string; approvedScope: string[]; approvalRef: Ref; customerConsent: string };
export type RepairCheck = { checkId: ID; method: string; observes: string; sourceRef: Ref };
export type RepairCheckpoint = { stage: string; completedTaskIds: ID[]; symptomVersion: string; pendingMonitoring: string[] };
export type RepairWorkReceipt = { receiptId: ID; taskId: ID; symptomVersion: string; plannedStep: string; disposition: "completed" | "partial" | "not-done" | "unknown"; verificationRefs: Ref[]; observedCheckIds: ID[] };
export type SymptomSeparation = { observations: string[]; hypotheses: string[]; unresolvedAttribution: string[]; isolationNeeded: boolean };
export type ApprovedRepairDiagnostic = { procedureId: ID; check: string; requiredSkill: string; expectedDistinction: string; stopCondition: string; route: "workshop" | "specialist" | "blocked" };
export type RepairerMatch = { taskId: ID; repairerId: ID; skillEvidence: Ref[]; supervised: boolean; slot: string };
export type RepairSequence = { steps: string[]; toolReservations: string[]; compatiblePartIds: ID[]; approvalRef: Ref; blockedChanges: string[]; route: "workshop" | "specialist" | "blocked" };
export type VerifiedRepairOutcome = { comparedChecks: string[]; verified: boolean; remainingIssue: string; recurrenceWatch: string[]; specialistReferral: string; scopedLesson: string };
export type RepairDispatchProposal = { requestId: ID; symptom: SymptomSeparation; diagnostic: ApprovedRepairDiagnostic; assignments: RepairerMatch[]; work: RepairSequence; outcome: VerifiedRepairOutcome; checkpoint: RepairCheckpoint; commitAllowed: boolean };
export type CredentialTaskFit = { taskId: ID; repairerId: ID; skillMatch: string; credentialEvidence: Ref[]; current: boolean; scopeCoversWork: boolean };
export type PartCompatibilityReview = { eligiblePartIds: ID[]; unsupportedPartIds: ID[]; missingEvidence: Ref[] };
export type ToolReservationReview = { reservedToolWindows: string[]; conflicts: string[]; tasksWithoutTool: ID[] };
export type RepairBudgetReview = { approvedWorkSteps: string[]; overBudgetChanges: string[]; reapprovalRequired: boolean };
export type CheckComparison = { verifiedChecks: string[]; missingChecks: string[]; conditionDifferences: string[] };
