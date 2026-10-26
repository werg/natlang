export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type ServiceNavigation = RecordMeta & WorkRecord & {
  applicantId: string;
  statedSituation: string;
  serviceCatalogId: string;
  coordinatorId: string;
  consentScope: string;
};
export type ServiceSituation = RecordMeta & WorkRecord & {
  caseId: string;
  statedNeeds: string[];
  knownFacts: string[];
  unknownFacts: string[];
  urgentRoutingFlag: string;
};
export type ServicePathway = RecordMeta & WorkRecord & {
  caseId: string;
  serviceId: string;
  potentialFit: string;
  requiredConditions: string[];
  unresolvedConditions: string[];
};
export type NavigationQuestion = RecordMeta & WorkRecord & {
  caseId: string;
  pathwayIds: string[];
  requestedFact: string;
  acceptableEvidence: string;
  purpose: string;
};
export type ApplicationTask = RecordMeta & WorkRecord & {
  caseId: string;
  pathwayId: string;
  requiredDocument: string;
  ownerId: string;
  completionCondition: string;
};
export type ServiceApplicationStatus = RecordMeta & WorkRecord & {
  caseId: string;
  pathwayId: string;
  submissionRef: string;
  agencyStatus: string;
  nextRequiredAction: string;
};
export type DraftServiceSituation = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    statedNeeds?: string[];
    knownFacts?: string[];
    unknownFacts?: string[];
    urgentRoutingFlag?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type NavigatorAgent = { id: string };
export type ServiceChange = { newEvidence: EvidenceItem[]; ruleUpdates: EvidenceItem[]; helperConsent: EvidenceItem[]; agencyReceipts: EvidenceItem[]; formExceptions: EvidenceItem[] };
export type NavigationDossier = { navigation: ServiceNavigation; situation: ServiceSituation[]; pathways: ServicePathway[]; questions: NavigationQuestion[]; tasks: ApplicationTask[]; statuses: ServiceApplicationStatus[]; catalogRules: EvidenceItem[]; evidence: EvidenceItem[]; checkpoint: NavigationCheckpoint };
export type EvidenceRoute = { condition: string; acceptedAlternatives: string[]; providedEvidenceIds: string[]; verificationState: 'verified'|'pending'|'missing'|'rejected'; ruleVersion: string; };
export type PathwayGraph = { pathways: ServicePathway[]; evidenceRoutes: EvidenceRoute[]; excludedRoutes: string[]; changedRuleIds: string[]; blockers: string[]; };
export type ApplicationTruth = { pathwayId: string; representedFacts: string[]; unsupportedFields: string[]; clarificationTasks: string[]; helperScope: string; };
export type ReceiptReconciliation = { statuses: ServiceApplicationStatus[]; receiptRefs: string[]; uncertainSubmissions: string[]; safeToResubmit: string[]; blockers: string[]; };
export type NavigationCheckpoint = { revision: string; situation: ServiceSituation[]; pathways: PathwayGraph; questions: NavigationQuestion[]; tasks: ApplicationTask[]; applicationTruths: ApplicationTruth[]; agencyReceipts: EvidenceItem[]; consentVersion: string; };
export type ApplicationWork = { tasks: ApplicationTask[]; truths: ApplicationTruth[]; blockers: string[]; };
export type NavigatorProposal = { application: 'Public Service Application Navigator'; status: 'ready-for-review'|'blocked'; situation: ServiceSituation[]; pathwayGraph: PathwayGraph; questions: NavigationQuestion[]; tasks: ApplicationTask[]; applicationTruths: ApplicationTruth[]; outcome: ReceiptReconciliation; blockers: string[]; actionBoundary: string; resumeState: NavigationCheckpoint; };
