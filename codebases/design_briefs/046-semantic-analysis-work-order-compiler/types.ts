export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type AnalysisRequest = RecordMeta & {
  question: string;
  datasetIds: ID[];
  requesterId: ID;
  decisionPurpose: string;
  resultAudience: string;
};
export type AnalysisMeaning = RecordMeta & {
  caseId: ID;
  population: string;
  semanticPredicates: string[];
  timeRelation: string;
  outputGrain: string;
  ambiguity: string;
};
export type DataSuitability = RecordMeta & {
  caseId: ID;
  meaningId: ID;
  requiredFields: string[];
  availableFields: string[];
  missingEvidence: string;
};
export type AnalysisWorkGraph = RecordMeta & {
  caseId: ID;
  nodeIds: ID[];
  semanticNodeIds: ID[];
  exactNodeIds: ID[];
  outputContract: string;
};
export type AnalysisSlice = RecordMeta & {
  caseId: ID;
  graphId: ID;
  inputManifestId: ID;
  classifiedIds: ID[];
  aggregateArtifactId: ID;
  coverageGap: string;
};
export type AnalysisAnswer = RecordMeta & {
  caseId: ID;
  sliceId: ID;
  answer: string;
  qualification: string;
  reproducibilityManifestId: ID;
};
export type DraftAnalysisMeaning = { values: { caseId?: ID; population?: string; semanticPredicates?: string[]; timeRelation?: string; outputGrain?: string; ambiguity?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type SemanticWorkOrderRequest = { requestId: ID; questionText: string; sourceEvidence: Evidence[]; schemaCatalog: string[]; recordRelationships: AnalysisJoin[]; accessRules: RecordAccessRule[]; requesterId: ID | null; requestAudience: string; approvedMeaning: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; requestedGroup: string[]; outputGrain: string; rubricVersion: string; changedEvidenceIds: ID[]; priorMeaning: AnalysisMeaning; priorRun: AnalysisSlice | null; materializedRows: MaterializedAnalysisRow[]; resultEvidence: Evidence[] };
export type AnalysisMeaningMap = { population: string; entities: string[]; predicates: string[]; timeRelation: string; grouping: string[]; outputGrain: string; alternateReadings: string[]; blockingQuestions: string[] };
export type RecordAccessRule = { recordClass: string; requesterId: ID; permitted: boolean; policyRef: ID };
export type AuthorizedAnalysisUniverse = { allowedRecordClasses: string[]; excludedClasses: string[]; discloseableExcludedCount: boolean };
export type FieldSupportMap = { mappings: { concept: string; field: string; semanticMatch: string; evidenceRefs: ID[]; accessible: boolean }[]; unsupportedConcepts: string[]; proxyWarnings: string[] };
export type AnalysisJoin = { leftNode: ID; rightNode: ID; key: string; cardinality: 'one-to-one' | 'many-to-one' | 'one-to-many' | 'unknown'; expectedGrain: string; evidenceRefs: ID[] };
export type AnalysisNode = { id: ID; kind: string; inputType: string; outputType: string; prerequisites: ID[]; sourceRefs: ID[] };
export type AnalysisEdge = { from: ID; to: ID; valueType: string };
export type AnalysisGraphPlan = { nodes: AnalysisNode[]; edges: AnalysisEdge[]; joins: AnalysisJoin[]; outputGrain: string; cycles: ID[]; typeErrors: string[] };
export type JoinGrainCheck = { allowed: boolean; repeatedEntityKey: string; fanout: string; deduplicationRule: string };
export type GraphTypeCheck = { acyclic: boolean; unresolvedNodeIds: ID[]; mismatchedEdges: ID[] };
export type MaterializedAnalysisRow = { rowId: ID; values: string[]; disposition: string; judgmentRubricVersion: string };
export type AnalysisExecutionAudit = { includedRows: ID[]; excludedRows: ID[]; inaccessibleRows: ID[]; missingRows: ID[]; unresolvedRows: ID[]; cardinalityMismatches: string[]; materializedValues: string[]; groupingChanged: boolean };
export type RubricReclassificationPlan = { previousRunId: ID | null; previousRubricVersion: string; affectedRowIds: ID[]; unaffectedJudgmentsReusable: ID[]; newRunRequired: boolean };
export type AnalysisAnswerProposal = { status: 'work-order-only' | 'review' | 'ready'; interpretation: string; numericalClaims: string[]; excludedCoverage: string[]; uncertainty: string[]; resultEvidenceRefs: ID[] };
export type SemanticWorkOrderProposal = { requestId: ID; meaning: AnalysisMeaningMap; suitability: FieldSupportMap; graph: AnalysisGraphPlan; execution: AnalysisExecutionAudit; answer: AnalysisAnswerProposal; commitAllowed: boolean };
export type AnalysisMaterializationReceipt = { receiptId: ID; requestId: ID; runId: ID; graphDigest: string; schemaSnapshotVersion: string; rubricVersion: string; resultArtifactId: ID; rows: MaterializedAnalysisRow[]; includedRowIds: ID[]; excludedRowIds: ID[]; inaccessibleRowIds: ID[]; missingRowIds: ID[]; cardinalityEvidence: ID[]; outputAudienceId: ID; audienceAuthorization: "authorized" | "denied" | "unknown"; evidenceRefs: ID[]; materializedAt: DateTime };
export type AnalysisRunRecord = { runId: ID; requestId: ID; graphDigest: string; schemaSnapshotVersion: string; rubricVersion: string; artifactId: ID; status: "current" | "held" | "needs-reclassification" | "unknown"; outputAudienceId: ID; evidenceRefs: ID[]; reasons: string[] };
export type AnalysisRunBook = { revision: number; currentRunId: ID | null; runs: AnalysisRunRecord[]; receipts: AnalysisMaterializationReceipt[]; consumedReceiptIds: ID[] };
export type AnalysisMaterializationAssessment = { publishable: boolean; status: "current" | "held" | "needs-reclassification" | "unknown"; reasons: string[]; protectedRowIds: ID[] };
export type AnalysisRunReconciliation = { book: AnalysisRunBook; outcome: "recorded" | "replayed" | "held"; resultVisible: boolean; nextSteps: string[] };
export type MaterializedRowCoverage = { complete: boolean; protectedRowIds: ID[]; reasons: string[] };
export type RubricLineageAssessment = { current: boolean; reclassificationRequired: boolean; reusableJudgments: ID[]; reasons: string[] };
export type AnalysisDisclosureAssessment = { status: "allowed" | "blocked" | "unknown"; reasons: string[] };
