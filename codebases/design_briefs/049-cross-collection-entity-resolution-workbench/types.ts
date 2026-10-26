export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: IdentityCandidate } };
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
export type ResolutionProject = RecordMeta & {
  collectionIds: ID[];
  entityKind: string;
  identityPolicyId: ID;
  stewardId: ID;
  intendedUse: string;
};
export type IdentityCandidate = RecordMeta & {
  caseId: ID;
  leftRecordId: ID;
  rightRecordId: ID;
  blockingReasons: string[];
  disqualifyingFacts: string[];
};
export type IdentityJudgment = RecordMeta & {
  caseId: ID;
  leftId: ID;
  rightId: ID;
  relation: 'same'|'different'|'related'|'unknown';
  decisiveEvidence: string;
};
export type IdentityCluster = RecordMeta & {
  caseId: ID;
  memberIds: ID[];
  cannotLinkIds: ID[];
  unresolvedPairIds: ID[];
  consistencyFinding: string;
};
export type CanonicalEntity = RecordMeta & {
  caseId: ID;
  memberIds: ID[];
  preferredName: string;
  attributeClaims: string[];
  unresolvedAttributes: string[];
};
export type IdentityRevision = RecordMeta & {
  caseId: ID;
  clusterId: ID;
  operation: 'merge'|'split'|'relabel'|'defer';
  affectedConsumerIds: ID[];
};
export type DraftIdentityCandidate = { values: { caseId?: ID; leftRecordId?: ID; rightRecordId?: ID; blockingReasons?: string[]; disqualifyingFacts?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultIdentityCandidate = { accepted: IdentityCandidate[]; unresolved: DraftIdentityCandidate[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftIdentityJudgment = { values: { caseId?: ID; leftId?: ID; rightId?: ID; relation?: 'same'|'different'|'related'|'unknown'; decisiveEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultIdentityJudgment = { accepted: IdentityJudgment[]; unresolved: DraftIdentityJudgment[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftIdentityCluster = { values: { caseId?: ID; memberIds?: ID[]; cannotLinkIds?: ID[]; unresolvedPairIds?: ID[]; consistencyFinding?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultIdentityCluster = { accepted: IdentityCluster[]; unresolved: DraftIdentityCluster[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftCanonicalEntity = { values: { caseId?: ID; memberIds?: ID[]; preferredName?: string; attributeClaims?: string[]; unresolvedAttributes?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultCanonicalEntity = { accepted: CanonicalEntity[]; unresolved: DraftCanonicalEntity[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftIdentityRevision = { values: { caseId?: ID; clusterId?: ID; operation?: 'merge'|'split'|'relabel'|'defer'; affectedConsumerIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultIdentityRevision = { accepted: IdentityRevision[]; unresolved: DraftIdentityRevision[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; constructCandidates: StageResultIdentityCandidate; judgePairs: StageResultIdentityJudgment; reconcileClusters: StageResultIdentityCluster; constructCanonical: StageResultCanonicalEntity; maintainIdentity: StageResultIdentityRevision; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { constructCandidates: StageResultIdentityCandidate; judgePairs: StageResultIdentityJudgment; reconcileClusters: StageResultIdentityCluster; constructCanonical: StageResultCanonicalEntity; maintainIdentity: StageResultIdentityRevision; };
export type EntityResolutionRequest = { resolutionId: ID; entityKind: string; sourceEvidence: Evidence[]; priorCandidate: IdentityCandidate; collectionAccessScopes: string[]; identityRules: string[]; knownCannotLinkPairs: string[]; recordValidityIntervals: string[]; fieldAuthorityRules: string[]; sourceRevocations: ID[]; downstreamContracts: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; }
export type IdentityPairCandidate = { leftRecordId: ID; rightRecordId: ID; blockingEvidence: string[]; sourceRefs: ID[]; accessScope: string; hardIncompatibilities: string[]; auditSample: boolean };
export type IdentityEvidenceStrength = { stableIdentifiers: string[]; nameEvidence: string[]; contextualEvidence: string[]; hardContradictions: string[]; unknownAttributes: string[]; timeIntervals: string[] };
export type PairIdentityJudgment = { leftRecordId: ID; rightRecordId: ID; relation: 'same' | 'different' | 'related' | 'unknown'; positiveEvidence: ID[]; negativeEvidence: ID[]; missingEvidence: string[]; relationshipKind: string; validityDecision: string };
export type ConstrainedIdentityClusters = { clusters: { clusterId: ID; memberIds: ID[]; acceptedEdges: string[] }[]; blockedMerges: string[]; reviewTriangles: string[] }
export type AttributedCanonicalView = { clusterId: ID; values: { field: string; value: string; sourceId: ID; effectiveFrom: string; effectiveTo: string; authorityRule: string }[]; aliases: string[]; disputedClaims: string[] }
export type IdentityCorrectionPlan = { priorGraphVersion: string; proposedVersion: string; changedEdges: string[]; reverseOperations: string[]; downstreamLinks: string[]; sourceRevocationEffects: string[]; promotionBlocked: string[] }
export type EntityResolutionProposal = { resolutionId: ID; candidates: IdentityPairCandidate[]; judgments: PairIdentityJudgment[]; clusters: ConstrainedIdentityClusters; canonical: AttributedCanonicalView[]; revision: IdentityCorrectionPlan; commitAllowed: boolean }
