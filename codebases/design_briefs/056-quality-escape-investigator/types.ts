export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DefectObservation } };
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
export type StageResultDefectObservation = { accepted: DefectObservation[]; unresolved: DraftDefectObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftQualityExposure = { values: { caseId?: ID; observationIds?: ID[]; affectedLotIds?: ID[]; downstreamItemIds?: ID[]; uncertainBoundary?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultQualityExposure = { accepted: QualityExposure[]; unresolved: DraftQualityExposure[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftQualityHypothesis = { values: { caseId?: ID; mechanism?: string; supportingRefs?: Ref[]; contradictingRefs?: Ref[]; discriminatingTest?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultQualityHypothesis = { accepted: QualityHypothesis[]; unresolved: DraftQualityHypothesis[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftQualityActionPlan = { values: { caseId?: ID; hypothesisIds?: ID[]; containmentTasks?: ID[]; correctiveTasks?: ID[]; releaseCriteria?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultQualityActionPlan = { accepted: QualityActionPlan[]; unresolved: DraftQualityActionPlan[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftQualityClosure = { values: { caseId?: ID; actionIds?: ID[]; verificationRefs?: Ref[]; residualRisk?: string; monitoringRule?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultQualityClosure = { accepted: QualityClosure[]; unresolved: DraftQualityClosure[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; normalizeDefects: StageResultDefectObservation; traceExposure: StageResultQualityExposure; compareCauses: StageResultQualityHypothesis; planCorrection: StageResultQualityActionPlan; verifyEffectiveness: StageResultQualityClosure; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { normalizeDefects: StageResultDefectObservation; traceExposure: StageResultQualityExposure; compareCauses: StageResultQualityHypothesis; planCorrection: StageResultQualityActionPlan; verifyEffectiveness: StageResultQualityClosure; };

export type QualityEscapeCase = { caseId: ID; observedDefects: DefectObservation[]; lotLineage: LotLineage[]; processHistories: ProcessHistory[]; inspectionMethods: InspectionMethod[]; holds: UnitHold[]; correctiveActions: CorrectiveActionRecord[]; evidence: Evidence[]; eventNote: string; priorClosure: QualityClosure | null };
export type LotLineage = { lotId: ID; parentLotId: ID | null; processId: ID; routeVersion: string; quantity: number; reworkRef: Ref | null };
export type ProcessHistory = { processId: ID; lotId: ID; routeVersion: string; operation: string; observedResult: string; operatorRef: Ref | null; evidenceRef: Ref };
export type InspectionMethod = { methodId: ID; revision: string; detectionThreshold: string; coverage: string; activeFrom: string; activeUntil: string | null };
export type UnitHold = { lotId: ID; unitIds: ID[]; scope: string; disposition: string; evidenceRef: Ref };
export type CorrectiveActionRecord = { actionId: ID; targetLotIds: ID[]; change: string; completed: boolean; verificationRefs: Ref[] };
export type EscapeTrace = { affectedLots: ID[]; reworkedBranches: ID[]; exposureBoundaries: string[]; preservedHistories: ID[] };
export type DefectRateComparison = { comparable: boolean; methodChanges: string[]; defectEvidence: string[]; detectionChange: string; rateClaim: string };
export type CausalReview = { mechanisms: string[]; supportingRefs: Ref[]; contraryRefs: Ref[]; discriminatingTests: string[]; blameInferenceBlocked: boolean };
export type ContainmentDisposition = { heldUnitIds: ID[]; releasableUnitIds: ID[]; unresolvedLotIds: ID[]; releaseBasis: string[] };
export type EffectivenessReview = { reopened: boolean; failedActionIds: ID[]; recurrenceEvidence: Ref[]; nextVerification: string; closurePermitted: boolean };
export type QualityEscapeProposal = { caseId: ID; trace: EscapeTrace; comparison: DefectRateComparison; causes: CausalReview; containment: ContainmentDisposition; effectiveness: EffectivenessReview; commitAllowed: boolean };
