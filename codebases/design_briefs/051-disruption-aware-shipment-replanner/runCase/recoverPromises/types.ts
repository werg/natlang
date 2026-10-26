export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DisruptionImpact } };
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
export type ShipmentDisruption = RecordMeta & {
  shipmentIds: ID[];
  disruptionDescription: string;
  coordinatorId: ID;
  planningWindow: string;
  priorityPolicyId: ID;
};
export type DisruptionImpact = RecordMeta & {
  caseId: ID;
  routeIds: ID[];
  affectedWindow: string;
  unavailableCapabilities: string[];
  uncertainty: string;
};
export type ShippingCommitment = RecordMeta & {
  caseId: ID;
  shipmentId: ID;
  promisedOutcome: string;
  deadline: DateTime;
  essentialConditions: string[];
};
export type ShippingAlternative = RecordMeta & {
  caseId: ID;
  shipmentIds: ID[];
  routeIds: ID[];
  substitutions: string[];
  changedPromises: string[];
};
export type ShippingAllocation = RecordMeta & {
  caseId: ID;
  alternativeIds: ID[];
  reservedCapacityIds: ID[];
  totalCost: Money;
  unresolvedDemand: string;
};
export type ShippingRecovery = RecordMeta & {
  caseId: ID;
  allocationId: ID;
  communicationIds: ID[];
  carrierReceiptIds: ID[];
  remainingRisk: string;
};
export type DraftDisruptionImpact = { values: { caseId?: ID; routeIds?: ID[]; affectedWindow?: string; unavailableCapabilities?: string[]; uncertainty?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDisruptionImpact = { accepted: DisruptionImpact[]; unresolved: DraftDisruptionImpact[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftShippingCommitment = { values: { caseId?: ID; shipmentId?: ID; promisedOutcome?: string; deadline?: DateTime; essentialConditions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultShippingCommitment = { accepted: ShippingCommitment[]; unresolved: DraftShippingCommitment[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftShippingAlternative = { values: { caseId?: ID; shipmentIds?: ID[]; routeIds?: ID[]; substitutions?: string[]; changedPromises?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultShippingAlternative = { accepted: ShippingAlternative[]; unresolved: DraftShippingAlternative[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftShippingAllocation = { values: { caseId?: ID; alternativeIds?: ID[]; reservedCapacityIds?: ID[]; totalCost?: Money; unresolvedDemand?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultShippingAllocation = { accepted: ShippingAllocation[]; unresolved: DraftShippingAllocation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftShippingRecovery = { values: { caseId?: ID; allocationId?: ID; communicationIds?: ID[]; carrierReceiptIds?: ID[]; remainingRisk?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultShippingRecovery = { accepted: ShippingRecovery[]; unresolved: DraftShippingRecovery[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretDisruption: StageResultDisruptionImpact; recoverPromises: StageResultShippingCommitment; constructAlternatives: StageResultShippingAlternative; allocateCapacity: StageResultShippingAllocation; coordinateRecovery: StageResultShippingRecovery; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretDisruption: StageResultDisruptionImpact; recoverPromises: StageResultShippingCommitment; constructAlternatives: StageResultShippingAlternative; allocateCapacity: StageResultShippingAllocation; coordinateRecovery: StageResultShippingRecovery; };
