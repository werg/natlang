export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: HarvestReadiness } };
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
export type HarvestRound = RecordMeta & {
  growerIds: ID[];
  cropLotIds: ID[];
  deliveryCommitmentIds: ID[];
  coordinatorId: ID;
  planningWindow: string;
};
export type HarvestReadiness = RecordMeta & {
  caseId: ID;
  lotId: ID;
  observedMaturity: string;
  qualityEvidence: Ref[];
  readinessWindow: string;
  uncertainty: string;
};
export type HarvestCommitment = RecordMeta & {
  caseId: ID;
  customerId: ID;
  requiredQuality: string;
  quantity: Quantity;
  deliveryWindow: string;
  substitutionsAllowed: string[];
};
export type HarvestMatch = RecordMeta & {
  caseId: ID;
  lotIds: ID[];
  commitmentId: ID;
  suitability: string;
  missingEvidence: string;
};
export type HarvestPlan = RecordMeta & {
  caseId: ID;
  matchIds: ID[];
  laborSlots: string[];
  equipmentIds: ID[];
  transportIds: ID[];
};
export type HarvestRevision = RecordMeta & {
  caseId: ID;
  changedLotIds: ID[];
  affectedCommitmentIds: ID[];
  proposedSubstitution: string;
  acceptanceRef: Ref;
};
export type DraftHarvestReadiness = { values: { caseId?: ID; lotId?: ID; observedMaturity?: string; qualityEvidence?: Ref[]; readinessWindow?: string; uncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHarvestReadiness = { accepted: HarvestReadiness[]; unresolved: DraftHarvestReadiness[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHarvestCommitment = { values: { caseId?: ID; customerId?: ID; requiredQuality?: string; quantity?: Quantity; deliveryWindow?: string; substitutionsAllowed?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHarvestCommitment = { accepted: HarvestCommitment[]; unresolved: DraftHarvestCommitment[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHarvestMatch = { values: { caseId?: ID; lotIds?: ID[]; commitmentId?: ID; suitability?: string; missingEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHarvestMatch = { accepted: HarvestMatch[]; unresolved: DraftHarvestMatch[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHarvestPlan = { values: { caseId?: ID; matchIds?: ID[]; laborSlots?: string[]; equipmentIds?: ID[]; transportIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHarvestPlan = { accepted: HarvestPlan[]; unresolved: DraftHarvestPlan[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHarvestRevision = { values: { caseId?: ID; changedLotIds?: ID[]; affectedCommitmentIds?: ID[]; proposedSubstitution?: string; acceptanceRef?: Ref }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHarvestRevision = { accepted: HarvestRevision[]; unresolved: DraftHarvestRevision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretReadiness: StageResultHarvestReadiness; recoverCommitments: StageResultHarvestCommitment; matchLots: StageResultHarvestMatch; scheduleResources: StageResultHarvestPlan; replanDelivery: StageResultHarvestRevision; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretReadiness: StageResultHarvestReadiness; recoverCommitments: StageResultHarvestCommitment; matchLots: StageResultHarvestMatch; scheduleResources: StageResultHarvestPlan; replanDelivery: StageResultHarvestRevision; };
