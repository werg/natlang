export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: CategoryFit } };
export type GoalContract = { outcome: string; exclusions: string[]; affectedEntities: string[]; acceptanceEvidence: string[]; clarificationNeeded: string[] };
export type PolicySelection = { selected: string[]; unresolvedPrecedence: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized" };
export type WorkGraph = { nodes: string[]; prerequisites: string[]; blockedQuestions: string[] };
export type EvidencePlan = { requiredEvidence: string[]; selectedIds: string[]; searchedUniverse: string[]; missingCoverage: string[] };
export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";
export type AuditResult = { verdict: Qualification; gate: string; reason: string; repairTargets: string[] };
export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };
export type Rejection = { id: string; reason: string; evidenceIds: string[] };
export type CoverageManifest = { eligible: string[]; processed: string[]; deferred: string[]; rejected: string[]; outOfScope: string[] };
export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };
export type ActionProposal = { kind: string; targetIds: string[]; reason: string; approvalRequired: boolean };
export type CheckResult = { check: string; verdict: Qualification; evidenceIds: string[] };
export type CollectionFinding = { acceptedRefs: string[]; unresolvedRefs: string[]; conflicts: string[]; coverageGaps: string[] };
export type QuestionBatch = { questions: string[]; authorizedRespondent: string; acceptableEvidence: string[] };
export type InvalidationProposal = { invalidated: string[]; preserved: string[]; unresolvedImpact: string[] };
export type ReviewPacket = { supportedDecisions: string[]; unresolvedAlternatives: string[]; actionScope: string[]; approvalBindings: string[] };
export type ViewModel = { title: string; status: "ready" | "blocked" | "review"; sections: string[]; uncertainties: string[] };
export type RevisionProposal = { target: string; change: string; contracts: string[]; expectedBenefit: string; regressionRisks: string[]; tests: string[] };

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
export type StageResultCategoryFit = { accepted: CategoryFit[]; unresolved: DraftCategoryFit[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftTaxonomyDefect = { values: { caseId?: ID; categoryIds?: ID[]; kind?: 'split'|'merge'|'rename'|'boundary'|'new'; evidenceIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultTaxonomyDefect = { accepted: TaxonomyDefect[]; unresolved: DraftTaxonomyDefect[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftCategoryRevision = { values: { caseId?: ID; affectedIds?: ID[]; newDefinitions?: string[]; relationMap?: string; preservedDistinctions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultCategoryRevision = { accepted: CategoryRevision[]; unresolved: DraftCategoryRevision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftTaxonomyImpact = { values: { caseId?: ID; revisionId?: ID; changedRecordIds?: ID[]; historicalBridge?: string; unresolvedIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultTaxonomyImpact = { accepted: TaxonomyImpact[]; unresolved: DraftTaxonomyImpact[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftTaxonomyRelease = { values: { caseId?: ID; revisionId?: ID; approvalRefs?: Ref[]; migrationTaskIds?: ID[]; rollbackVersion?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultTaxonomyRelease = { accepted: TaxonomyRelease[]; unresolved: DraftTaxonomyRelease[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; assessFit: StageResultCategoryFit; diagnoseStructure: StageResultTaxonomyDefect; designCategoryRevision: StageResultCategoryRevision; simulateReclassification: StageResultTaxonomyImpact; promoteTaxonomy: StageResultTaxonomyRelease; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { assessFit: StageResultCategoryFit; diagnoseStructure: StageResultTaxonomyDefect; designCategoryRevision: StageResultCategoryRevision; simulateReclassification: StageResultTaxonomyImpact; promoteTaxonomy: StageResultTaxonomyRelease; };
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
