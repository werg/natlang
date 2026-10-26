export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DonationOffer } };
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
export type DonationRound = RecordMeta & {
  offerIds: ID[];
  recipientNeedIds: ID[];
  coordinatorId: ID;
  allocationRuleId: ID;
  distributionWindow: string;
};
export type DonationOffer = RecordMeta & {
  caseId: ID;
  donorId: ID;
  itemDescription: string;
  quantity: Quantity;
  condition: string;
  availabilityWindow: string;
};
export type RecipientNeed = RecordMeta & {
  caseId: ID;
  recipientId: ID;
  desiredOutcome: string;
  essentialConstraints: string[];
  acceptableSubstitutions: string[];
};
export type DonationBundle = RecordMeta & {
  caseId: ID;
  offerIds: ID[];
  recipientId: ID;
  intendedUse: string;
  missingComponents: string[];
};
export type DonationAllocation = RecordMeta & {
  caseId: ID;
  bundleIds: ID[];
  reservedQuantities: Quantity[];
  ruleBasis: string;
  unmetNeedIds: ID[];
};
export type DonationDelivery = RecordMeta & {
  caseId: ID;
  allocationId: ID;
  logisticsTaskIds: ID[];
  acceptanceRef: Ref;
  declinedItems: ID[];
};
export type DraftDonationOffer = { values: { caseId?: ID; donorId?: ID; itemDescription?: string; quantity?: Quantity; condition?: string; availabilityWindow?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDonationOffer = { accepted: DonationOffer[]; unresolved: DraftDonationOffer[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftRecipientNeed = { values: { caseId?: ID; recipientId?: ID; desiredOutcome?: string; essentialConstraints?: string[]; acceptableSubstitutions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultRecipientNeed = { accepted: RecipientNeed[]; unresolved: DraftRecipientNeed[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDonationBundle = { values: { caseId?: ID; offerIds?: ID[]; recipientId?: ID; intendedUse?: string; missingComponents?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDonationBundle = { accepted: DonationBundle[]; unresolved: DraftDonationBundle[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDonationAllocation = { values: { caseId?: ID; bundleIds?: ID[]; reservedQuantities?: Quantity[]; ruleBasis?: string; unmetNeedIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDonationAllocation = { accepted: DonationAllocation[]; unresolved: DraftDonationAllocation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDonationDelivery = { values: { caseId?: ID; allocationId?: ID; logisticsTaskIds?: ID[]; acceptanceRef?: Ref; declinedItems?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDonationDelivery = { accepted: DonationDelivery[]; unresolved: DraftDonationDelivery[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; normalizeDonations: StageResultDonationOffer; interpretNeeds: StageResultRecipientNeed; constructBundles: StageResultDonationBundle; applyAllocation: StageResultDonationAllocation; coordinateDelivery: StageResultDonationDelivery; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { normalizeDonations: StageResultDonationOffer; interpretNeeds: StageResultRecipientNeed; constructBundles: StageResultDonationBundle; applyAllocation: StageResultDonationAllocation; coordinateDelivery: StageResultDonationDelivery; };
