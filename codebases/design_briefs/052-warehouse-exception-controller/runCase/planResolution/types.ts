export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: StockObservation } };
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
export type WarehouseException = RecordMeta & {
  warehouseId: ID;
  itemRefs: ID[];
  reportedDiscrepancy: string;
  operatorId: ID;
  inventorySnapshotId: ID;
};
export type StockObservation = RecordMeta & {
  caseId: ID;
  locationId: ID;
  observedDescription: string;
  serial: string;
  quantity: Quantity;
  condition: string;
};
export type StockIdentityFinding = RecordMeta & {
  caseId: ID;
  observationId: ID;
  candidateItemIds: ID[];
  verdict: string;
  missingEvidence: string;
};
export type WarehouseFinding = RecordMeta & {
  caseId: ID;
  itemId: ID;
  kind: 'misplaced'|'mislabeled'|'damaged'|'missing'|'pending';
  affectedQuantity: Quantity;
};
export type WarehouseWorkOrder = RecordMeta & {
  caseId: ID;
  findingId: ID;
  task: string;
  targetLocationId: ID;
  verificationRequired: string;
};
export type InventoryResolution = RecordMeta & {
  caseId: ID;
  workOrderIds: ID[];
  movementIds: ID[];
  adjustmentIds: ID[];
  residualDiscrepancy: string;
};
export type DraftStockObservation = { values: { caseId?: ID; locationId?: ID; observedDescription?: string; serial?: string; quantity?: Quantity; condition?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultStockObservation = { accepted: StockObservation[]; unresolved: DraftStockObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftStockIdentityFinding = { values: { caseId?: ID; observationId?: ID; candidateItemIds?: ID[]; verdict?: string; missingEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultStockIdentityFinding = { accepted: StockIdentityFinding[]; unresolved: DraftStockIdentityFinding[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftWarehouseFinding = { values: { caseId?: ID; itemId?: ID; kind?: 'misplaced'|'mislabeled'|'damaged'|'missing'|'pending'; affectedQuantity?: Quantity }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultWarehouseFinding = { accepted: WarehouseFinding[]; unresolved: DraftWarehouseFinding[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftWarehouseWorkOrder = { values: { caseId?: ID; findingId?: ID; task?: string; targetLocationId?: ID; verificationRequired?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultWarehouseWorkOrder = { accepted: WarehouseWorkOrder[]; unresolved: DraftWarehouseWorkOrder[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftInventoryResolution = { values: { caseId?: ID; workOrderIds?: ID[]; movementIds?: ID[]; adjustmentIds?: ID[]; residualDiscrepancy?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultInventoryResolution = { accepted: InventoryResolution[]; unresolved: DraftInventoryResolution[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; reconstructStock: StageResultStockObservation; matchIdentity: StageResultStockIdentityFinding; classifyException: StageResultWarehouseFinding; planResolution: StageResultWarehouseWorkOrder; reconcileInventory: StageResultInventoryResolution; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { reconstructStock: StageResultStockObservation; matchIdentity: StageResultStockIdentityFinding; classifyException: StageResultWarehouseFinding; planResolution: StageResultWarehouseWorkOrder; reconcileInventory: StageResultInventoryResolution; };
