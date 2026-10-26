export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type AppointmentPreparation = RecordMeta & WorkRecord & {
  patientId: string;
  appointmentId: string;
  approvedInstructionId: string;
  coordinatorId: string;
  communicationPreferences: string[];
};
export type PreparationRequirement = RecordMeta & WorkRecord & {
  caseId: string;
  instructionRef: string;
  requiredAction: string;
  timingRule: string;
  exception: string;
};
export type PreparationContext = RecordMeta & WorkRecord & {
  caseId: string;
  statedConstraint: string;
  communicationNeed: string;
  authorizedRepresentativeId: string;
  unresolvedQuestion: string;
};
export type PreparationChecklist = RecordMeta & WorkRecord & {
  caseId: string;
  requirementIds: string[];
  accessibleText: string[];
  exactTimingArtifactId: string;
  staffQuestionIds: string[];
};
export type PreparationQuestion = RecordMeta & WorkRecord & {
  caseId: string;
  requirementId: string;
  patientQuestion: string;
  responsibleServiceId: string;
  responseRef: string;
};
export type PreparationStatus = RecordMeta & WorkRecord & {
  caseId: string;
  checklistId: string;
  acknowledgedItems: string[];
  unresolvedItems: string[];
  contactOutcome: string;
};
export type DraftPreparationRequirement = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    instructionRef?: string;
    requiredAction?: string;
    timingRule?: string;
    exception?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type PreparationCoordinator = { id: string };
export type AppointmentChange = { instructionUpdates: EvidenceItem[]; appointmentUpdates: EvidenceItem[]; patientReports: EvidenceItem[]; translationReviews: EvidenceItem[]; representativeConsent: EvidenceItem[]; deliveries: EvidenceItem[] };
export type PreparationDossier = { appointment: AppointmentPreparation; requirements: PreparationRequirement[]; contexts: PreparationContext[]; checklists: PreparationChecklist[]; questions: PreparationQuestion[]; statuses: PreparationStatus[]; evidence: EvidenceItem[]; checkpoint: PreparationCheckpoint };
export type RelativeReminder = { requirementId: string; instructionVersion: string; appointmentVersion: string; eventTime: string; offsetMinutes: string; reminderTime: string; sourceIds: string[]; };
export type InstructionConflict = { statements: string[]; sourceIds: string[]; staffQuestion: string; };
export type DeliveryLedger = { channel: string; recipientId: string; consentVersion: string; state: 'pending'|'delivered'|'undelivered'|'acknowledged'|'unknown'; attemptRefs: string[]; fallbackChannel: string; blockers: string[]; };
export type PreparationCheckpoint = { revision: string; appointment: AppointmentPreparation; instructionVersion: string; appointmentVersion: string; reminders: RelativeReminder[]; checklist: PreparationChecklist[]; conflicts: InstructionConflict[]; translationReviewRefs: string[]; delivery: DeliveryLedger[]; acknowledgments: string[]; };
export type PreparationProposal = { application: 'Appointment Preparation Service'; status: 'ready-for-review'|'blocked'; requirements: PreparationRequirement[]; reminders: RelativeReminder[]; checklist: PreparationChecklist[]; questions: PreparationQuestion[]; conflicts: InstructionConflict[]; translations: string[]; deliveries: DeliveryLedger[]; blockers: string[]; actionBoundary: string; resumeState: PreparationCheckpoint; };

export type ChecklistCompilation = { checklists: PreparationChecklist[]; reminders: RelativeReminder[]; blockers: string[]; };
