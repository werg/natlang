export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type QualityInvestigation = RecordMeta & {
  productIds: ID[];
  lotIds: ID[];
  reportedDefect: string;
  qualityOwnerId: ID;
  investigationScope: string;
};
export type DefectObservation = RecordMeta & {
  caseId: ID;
  productId: ID;
  lotId: ID;
  symptom: string;
  observedAt: DateTime;
  inspectionContext: string;
};
export type QualityExposure = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  affectedLotIds: ID[];
  downstreamItemIds: ID[];
  uncertainBoundary: string;
};
export type QualityHypothesis = RecordMeta & {
  caseId: ID;
  mechanism: string;
  supportingRefs: Ref[];
  contradictingRefs: Ref[];
  discriminatingTest: string;
};
export type QualityActionPlan = RecordMeta & {
  caseId: ID;
  hypothesisIds: ID[];
  containmentTasks: ID[];
  correctiveTasks: ID[];
  releaseCriteria: string[];
};
export type QualityClosure = RecordMeta & {
  caseId: ID;
  actionIds: ID[];
  verificationRefs: Ref[];
  residualRisk: string;
  monitoringRule: string;
};
export type DraftDefectObservation = { values: { caseId?: ID; productId?: ID; lotId?: ID; symptom?: string; observedAt?: DateTime; inspectionContext?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type QualityEscapeCase = { caseId: ID; observedDefects: DefectObservation[]; lotLineage: LotLineage[]; processHistories: ProcessHistory[]; inspectionMethods: InspectionMethod[]; holds: UnitHold[]; correctiveActions: CorrectiveActionRecord[]; evidence: Evidence[]; eventNote: string; priorClosure: QualityClosure | null };
export type LotLineage = { lotId: ID; parentLotId: ID | null; processId: ID; routeVersion: string; quantity: number; reworkRef: Ref | null };
export type ProcessHistory = { processId: ID; lotId: ID; routeVersion: string; operation: string; observedResult: string; operatorRef: Ref | null; evidenceRef: Ref };
export type InspectionMethod = { methodId: ID; revision: string; detectionThreshold: string; coverage: string; activeFrom: string; activeUntil: string | null };
export type UnitHold = { lotId: ID; unitIds: ID[]; scope: string; disposition: string; evidenceRef: Ref };
export type CorrectiveActionRecord = { actionId: ID; targetLotIds: ID[]; change: string; completed: boolean; verificationRefs: Ref[] };
export type EscapeTrace = { affectedLots: ID[]; reworkedBranches: ID[]; exposureBoundaries: string[]; preservedHistories: ID[]; invalidEdges: string[] };
export type DefectRateComparison = { comparable: boolean; methodChanges: string[]; sampleFrame: SampleFrameReview; defectEvidence: string[]; detectionChange: string; rateClaim: string };
export type CausalReview = { mechanisms: string[]; supportingRefs: Ref[]; contraryRefs: Ref[]; discriminatingTests: string[]; blameInferenceBlocked: boolean; rivalTests: MechanismTest[] };
export type ContainmentDisposition = { heldUnitIds: ID[]; releasableUnitIds: ID[]; unresolvedLotIds: ID[]; releaseBasis: string[] };
export type EffectivenessReview = { reopened: boolean; failedActionIds: ID[]; recurrenceEvidence: Ref[]; nextVerification: string; closurePermitted: boolean };
export type QualityEscapeProposal = { caseId: ID; trace: EscapeTrace; comparison: DefectRateComparison; causes: CausalReview; containment: ContainmentDisposition; effectiveness: EffectivenessReview; commitAllowed: boolean };

export type GenealogyEdgeAudit = { verifiedEdges: string[]; invalidEdges: string[]; missingParentIds: ID[] };
export type SampleFrameReview = { denominators: string[]; methodVersions: string[]; seasonalWindows: string[]; comparablePairs: string[]; incomparablePairs: string[] };
export type MechanismSet = { mechanisms: string[]; predictedPatterns: string[]; discriminatingTests: string[] };
export type MechanismTest = { mechanism: string; supportingRefs: Ref[]; contraryRefs: Ref[]; predictionsMatched: string[]; remainingUnknowns: string[] };
export type QualityLotDispositionReceipt = { receiptId: ID; caseId: ID; lotId: ID; unitIds: ID[]; requestedDisposition: "hold" | "release"; observedDisposition: "held" | "released" | "partial" | "unknown"; physicalControl: string; evidenceRefs: Ref[]; occurredAt: DateTime };
export type QualityLotState = { lotId: ID; unitIds: ID[]; status: "held" | "released" | "released-unverified" | "unknown"; receiptIds: ID[]; blockers: string[] };
export type QualityDispositionBook = { revision: number; lots: QualityLotState[]; receipts: QualityLotDispositionReceipt[]; consumedReceiptIds: ID[] };
export type QualityLotAssessment = { safeToRelease: boolean; reasons: string[]; uncoveredUnitIds: ID[] };
export type QualityLotReconciliation = { book: QualityDispositionBook; outcome: "held" | "released" | "released-unverified" | "partial" | "unknown" | "replayed"; containmentTasks: string[] };
