export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type TaxonomyChange = RecordMeta & {
  taxonomyId: ID;
  incomingCollectionId: ID;
  stewardId: ID;
  intendedUse: string;
  baselineVersion: string;
};
export type CategoryFit = RecordMeta & {
  caseId: ID;
  recordId: ID;
  candidateCategoryIds: ID[];
  fit: 'clear'|'overlap'|'none'|'uncertain';
  boundaryIssue: string;
};
export type TaxonomyDefect = RecordMeta & {
  caseId: ID;
  categoryIds: ID[];
  kind: 'split'|'merge'|'rename'|'boundary'|'new';
  evidenceIds: ID[];
};
export type CategoryRevision = RecordMeta & {
  caseId: ID;
  affectedIds: ID[];
  newDefinitions: string[];
  relationMap: string;
  preservedDistinctions: string[];
};
export type TaxonomyImpact = RecordMeta & {
  caseId: ID;
  revisionId: ID;
  changedRecordIds: ID[];
  historicalBridge: string;
  unresolvedIds: ID[];
};
export type TaxonomyRelease = RecordMeta & {
  caseId: ID;
  revisionId: ID;
  approvalRefs: Ref[];
  migrationTaskIds: ID[];
  rollbackVersion: string;
};
export type DraftCategoryFit = { values: { caseId?: ID; recordId?: ID; candidateCategoryIds?: ID[]; fit?: 'clear'|'overlap'|'none'|'uncertain'; boundaryIssue?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type TaxonomyEvolutionRequest = { taxonomyId: ID; sourceEvidence: Evidence[]; priorFit: CategoryFit; baselineVersion: string; candidateVersion: string; currentDefinitions: string[]; positiveAndNegativeExamples: string[]; stewardBranches: TaxonomyStewardBranch[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; rareConceptImportanceEvidence: Evidence[]; downstreamLabelContracts: string[]; historicalRecords: TaxonomyRecord[]; changedRecordIds: ID[]; }
export type VersionedCategoryFit = { recordId: ID; categoryIds: ID[]; definitionVersion: string; fit: 'clear' | 'overlap' | 'none' | 'uncertain'; boundaryEvidence: string[]; sourceRefs: ID[] };
export type TaxonomyStructureDiagnosis = { kind: 'guidance' | 'boundary' | 'split' | 'merge' | 'rename' | 'new-concept'; affectedCategoryIds: ID[]; informativeExamples: string[]; counterexamples: string[]; decisiveRequirement: string; status: 'supported' | 'provisional' };
export type TaxonomyStewardBranch = { stewardId: ID; branchVersion: string; inclusionExamples: string[]; exclusionExamples: string[] };
export type StewardBoundaryConflict = { branchVersions: string[]; contradictoryExamples: string[]; mergeBlocked: boolean };
export type TaxonomyDefinitionRevision = { version: string; definitions: string[]; inclusionRules: string[]; exclusionRules: string[]; categoryIdsPreserved: ID[]; newCategoryIds: ID[]; historicalRelationMap: string; cyclicRelations: ID[]; contradictoryBoundaries: string[] };
export type TaxonomyRecord = { recordId: ID; categoryIds: ID[]; sourceVersion: string; evidenceRefs: ID[] };
export type TaxonomyRecordImpact = { changedRecordIds: ID[]; unchangedRecordIds: ID[]; ambiguousHistoricalBridges: ID[]; coarseHistoricalSeries: string[] };
export type TaxonomyConsumerCompatibility = { consumerId: ID; acceptedLabelCount: string; candidateLabelCount: string; compatible: boolean; requiredAdapter: string };
export type TaxonomyPromotionDecision = { status: 'blocked' | 'review' | 'ready'; version: string; approvalsNeeded: string[]; invalidatedRecordIds: ID[]; historicalDefinitionsRetained: boolean };
export type TaxonomyEvolutionProposal = { taxonomyId: ID; fit: VersionedCategoryFit[]; defect: TaxonomyStructureDiagnosis; revision: TaxonomyDefinitionRevision; impact: TaxonomyRecordImpact; compatibility: TaxonomyConsumerCompatibility[]; release: TaxonomyPromotionDecision; commitAllowed: boolean };
export type TaxonomyPromotionReceipt = { receiptId: ID; taxonomyId: ID; candidateVersion: string; definitionDigest: string; outcome: "activated" | "partial" | "unknown" | "reversed"; replayedRecordIds: ID[]; unresolvedRecordIds: ID[]; acceptedConsumerIds: ID[]; priorVersion: string | null; evidenceRefs: ID[]; reversalEvidenceRefs: ID[] };
export type TaxonomyReleaseBook = { revision: number; activeVersion: string | null; receipts: TaxonomyPromotionReceipt[]; consumedReceiptIds: ID[] };
export type TaxonomyReleaseReconciliation = { book: TaxonomyReleaseBook; outcome: "activated" | "partial" | "unknown" | "reversed" | "held" | "replayed"; unresolvedRecordIds: ID[]; incompatibleConsumerIds: ID[]; followUp: string[] };
export type TaxonomyActivationAssessment = { ready: boolean; reasons: string[]; unresolvedRecordIds: ID[]; incompatibleConsumerIds: ID[] };
