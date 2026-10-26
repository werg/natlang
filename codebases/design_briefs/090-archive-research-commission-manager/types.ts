export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type ArchiveCommission = RecordMeta & WorkRecord & {
  researchQuestion: string;
  requesterId: string;
  collectionIds: string[];
  researcherId: string;
  effortBudget: number;
};
export type ResearchCommissionScope = RecordMeta & WorkRecord & {
  caseId: string;
  subquestions: string[];
  intendedDeliverable: string;
  excludedClaims: string[];
  stoppingRule: string;
};
export type ArchiveSearchPlan = RecordMeta & WorkRecord & {
  caseId: string;
  questionIds: string[];
  collectionIds: string[];
  searchTerms: string[];
  coverageGaps: string[];
};
export type ArchiveLead = RecordMeta & WorkRecord & {
  caseId: string;
  sourceRef: string;
  relevantQuestion: string;
  newSearchTask: string;
  reliabilityLimit: string;
};
export type ArchiveFinding = RecordMeta & WorkRecord & {
  caseId: string;
  question: string;
  supportingRefs: string[];
  contradictoryRefs: string[];
  conclusion: string;
};
export type ResearchDossier = RecordMeta & WorkRecord & {
  caseId: string;
  findingIds: string[];
  searchedManifestId: string;
  remainingLeads: string[];
  rightsNotes: string[];
};
export type DraftResearchCommissionScope = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    subquestions?: string[];
    intendedDeliverable?: string;
    excludedClaims?: string[];
    stoppingRule?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type CommissionRequest = { id:string; researchQuestion:string; requesterId:string; researcherId:string; collectionIds:string[]; effortBudget:number; deliverable:string; exclusions:string[]; };
export type ArchiveSource = { id:string; collectionId:string; title:string; description:string; access:'open'|'closed'|'restricted'; revision:string; citation:string; rights:string[]; linkedSourceIds:string[]; };
export type SearchObservation = { id:string; collectionId:string; query:string; result:'hit'|'no-hit'|'closed'|'not-searched'; sourceIds:string[]; reason:string; };
export type ArchiveSearchCoverage={collectionId:string;question:string;access:'open'|'closed'|'restricted'|'missing';queryVariants:string[];matchingObservationIds:string[];coverage:'observed-hit'|'observed-no-hit'|'closed'|'restricted'|'not-searched'|'partial';reason:string;};
export type FollowupLead = { id:string; questionId:string; sourceId:string; rationale:string; missingSource:boolean; estimatedEffort:number; state:'open'|'pursued'|'stopped'|'unavailable'; };
export type SupportedClaim = { id:string; questionId:string; statement:string; evidenceIds:string[]; counterEvidenceIds:string[]; inferenceLimit:string; certainty:'direct'|'supported'|'contested'|'unknown'; };
export type ResearchCheckpoint = { requestId:string; questionIds:string[]; manifest:SearchObservation[]; leads:FollowupLead[]; claims:SupportedClaim[]; consumedEffort:number; sourceRevisions:string[]; };
export type CommissionDossier = { requestId:string; subquestions:string[]; claims:SupportedClaim[]; searchManifest:SearchObservation[]; searchCoverage:ArchiveSearchCoverage[]; remainingLeads:FollowupLead[]; coverageLimits:string[]; restrictedSources:string[]; interpretationNotes:string[]; checkpoint:ResearchCheckpoint; status:'bounded'|'extension-needed'|'incomplete'; };

export type ArchiveCommissionBinding = { request: CommissionRequest; evidenceIds: string[]; blockers: string[] };
