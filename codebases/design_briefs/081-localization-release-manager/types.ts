export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type LocalizationRelease = RecordMeta & WorkRecord & {
  sourceReleaseId: string;
  localeIds: string[];
  contentIds: string[];
  managerId: string;
  glossaryVersion: string;
};
export type LocalizationSegment = RecordMeta & WorkRecord & {
  caseId: string;
  sourceRef: string;
  contextKey: string;
  locale: string;
  protectedTokens: string[];
  semanticPurpose: string;
};
export type LocaleTermDecision = RecordMeta & WorkRecord & {
  caseId: string;
  segmentId: string;
  conceptId: string;
  approvedTerm: string;
  grammaticalContext: string;
  exception: string;
};
export type LocalizedSegment = RecordMeta & WorkRecord & {
  caseId: string;
  segmentId: string;
  translatedText: string;
  adaptationRationale: string;
  unresolvedIssue: string;
};
export type LocaleQualityFinding = RecordMeta & WorkRecord & {
  caseId: string;
  segmentIds: string[];
  issueKind: string;
  exactCheckIds: string[];
  reviewerFinding: string;
};
export type LocaleReleasePacket = RecordMeta & WorkRecord & {
  caseId: string;
  segmentIds: string[];
  sourceHash: string;
  reviewRefs: string[];
  fallbackPolicy: string;
};

export type LocaleSourceChange = { instruction: string; releases: LocalizationRelease[]; sourceEvidence: EvidenceItem[]; glossaryEvidence: EvidenceItem[]; expectedContentIds: string[]; releaseManager: string; previouslyCommittedIds: string[]; changedSources: EvidenceItem[]; structuralReceipts?: LocaleStructureReceipt[] };
export type ContextSegment = { key: string; contentId: string; contextId: string; role: string; sourceText: string; sourceVersion: string; purpose: string; protectedTokens: string[]; conceptIds: string[]; senseIds: string[]; journeyIds: string[]; questions: string[] };
export type ConceptLocaleTerm = { conceptId: string; senseId: string; locale: string; grammaticalRole: string; preferredForm: string | null; permittedVariants: string[]; glossaryVersion: string; authorityEvidenceIds: string[]; conflicts: string[] };
export type LocaleCandidate = { key: string; locale: string; candidateId: string; sourceVersion: string; contextVersion: string; glossaryVersion: string; text: string; meaningQuestions: string[]; usedConceptIds: string[]; usedSenseIds: string[] };
export type LocaleStructureReceipt = { receiptId: string; candidateId: string; sourceVersion: string; validatorVersion: string; placeholderIdentityOk: boolean; markupOk: boolean; lengthOk: boolean; pluralFormsOk: boolean; linksOk: boolean; renderedArtifactId: string };
export type LocaleFidelityReview = { candidateId: string; meaningPreserved: boolean; structuralReceiptIds: string[]; materialDefects: string[]; harmlessVariants: string[]; reviewerEvidenceIds: string[] };
export type LocaleBundle = { id: string; locale: string; sourceReleaseId: string; candidateIds: string[]; journeyIds: string[]; declaredFallbacks: string[]; approvalId: string | null; actualReleaseReceiptId: string | null; blockers: string[] };
export type LocaleEditionLedger = { revision: number; segments: ContextSegment[]; terminology: ConceptLocaleTerm[]; candidates: LocaleCandidate[]; reviews: LocaleFidelityReview[]; bundles: LocaleBundle[]; staleCandidateIds: string[] };
export type LocaleEditionReview = { localeLedger: LocaleEditionLedger; proposedBundles: LocaleBundle[]; terminologyQuestions: string[]; invalidatedCandidateIds: string[]; retainedCandidateIds: string[]; releaseBlocks: string[] };

export type LocaleInvalidation = { segmentKeys: string[]; candidateIds: string[] };
