export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };
export type DraftLocalizationSegment = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    sourceRef?: string;
    contextKey?: string;
    locale?: string;
    protectedTokens?: string[];
    semanticPurpose?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
