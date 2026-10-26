export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type ExhibitionProject = RecordMeta & WorkRecord & {
  collectionIds: string[];
  exhibitionBrief: string;
  audience: string;
  curatorId: string;
  venueConstraintIds: string[];
};
export type CuratorialEvidence = RecordMeta & WorkRecord & {
  caseId: string;
  artifactId: string;
  supportedInterpretation: string;
  provenanceLimit: string;
  displayConstraint: string;
};
export type ExhibitionArgument = RecordMeta & WorkRecord & {
  caseId: string;
  theme: string;
  centralQuestion: string;
  artifactIds: string[];
  competingInterpretation: string;
};
export type ExhibitionZone = RecordMeta & WorkRecord & {
  caseId: string;
  argumentId: string;
  artifactIds: string[];
  visitorTask: string;
  spatialRequirements: string[];
};
export type ExhibitionLabel = RecordMeta & WorkRecord & {
  caseId: string;
  artifactId: string;
  text: string;
  claimRefs: string[];
  audienceLayer: string;
};
export type ExhibitionRelease = RecordMeta & WorkRecord & {
  caseId: string;
  zoneIds: string[];
  labelIds: string[];
  rightsReviewRefs: string[];
  unresolvedConditions: string[];
};
export type DraftCuratorialEvidence = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    artifactId?: string;
    supportedInterpretation?: string;
    provenanceLimit?: string;
    displayConstraint?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type ExhibitionEditor = { id: string };
export type ExhibitionChange = { loanUpdates: EvidenceItem[]; conservationUpdates: EvidenceItem[]; communityRestrictions: EvidenceItem[]; visitorPathTests: EvidenceItem[]; labelFeedback: EvidenceItem[]; onlineRenditions: EvidenceItem[] };
export type ExhibitionDossier = { project: ExhibitionProject; evidence: CuratorialEvidence[]; arguments: ExhibitionArgument[]; zones: ExhibitionZone[]; labels: ExhibitionLabel[]; releases: ExhibitionRelease[]; sources: EvidenceItem[]; checkpoint: ExhibitionCheckpoint };
export type ObjectConstraint = { artifactId: string; loanState: string; provenanceLimit: string; permittedLux: string; durationLimit: string; communityRestriction: string; displayEvidenceIds: string[]; derivativeRestrictions: string[]; };
export type NarrativeSupport = { argumentId: string; claim: string; supportingArtifactIds: string[]; withdrawnArtifactIds: string[]; weakenedClaims: string[]; };
export type VisitorContextEdge = { fromZoneId: string; toZoneId: string; requiredContext: string[]; repeatedInLaterLabel: boolean; evidenceIds: string[]; };
export type ExhibitionCheckpoint = { revision: string; project: ExhibitionProject; objectConstraints: ObjectConstraint[]; arguments: ExhibitionArgument[]; narrativeSupport: NarrativeSupport[]; zones: ExhibitionZone[]; contextEdges: VisitorContextEdge[]; labels: ExhibitionLabel[]; release: ExhibitionRelease | null; invalidatedPlacements: string[]; derivativeRestrictionRefs: string[]; };
export type CuratorialProposal = { application: 'Museum Exhibition Curator'; status: 'ready-for-review'|'blocked'; objectConstraints: ObjectConstraint[]; arguments: ExhibitionArgument[]; narrativeSupport: NarrativeSupport[]; zones: ExhibitionZone[]; labels: ExhibitionLabel[]; contextEdges: VisitorContextEdge[]; release: ExhibitionRelease; blockers: string[]; actionBoundary: string; resumeState: ExhibitionCheckpoint; };
