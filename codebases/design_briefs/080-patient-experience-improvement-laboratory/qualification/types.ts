export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };
export type DraftExperienceEpisode = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    serviceId?: string;
    administrativeJourney?: string;
    reportedObstacle?: string;
    consequence?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
