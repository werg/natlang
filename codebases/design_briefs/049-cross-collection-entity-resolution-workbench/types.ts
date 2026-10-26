export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type EntityResolutionRequest = { resolutionId: ID; entityKind: string; sourceEvidence: Evidence[]; priorCandidate: IdentityCandidate; collectionAccessScopes: string[]; identityRules: string[]; knownCannotLinkPairs: string[]; recordValidityIntervals: string[]; fieldAuthorityRules: string[]; sourceRevocations: ID[]; downstreamContracts: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; }
export type IdentityPairCandidate = { leftRecordId: ID; rightRecordId: ID; blockingEvidence: string[]; sourceRefs: ID[]; accessScope: string; hardIncompatibilities: string[]; auditSample: boolean };
export type IdentityEvidenceStrength = { stableIdentifiers: string[]; nameEvidence: string[]; contextualEvidence: string[]; hardContradictions: string[]; unknownAttributes: string[]; timeIntervals: string[] };
export type PairIdentityJudgment = { leftRecordId: ID; rightRecordId: ID; relation: 'same' | 'different' | 'related' | 'unknown'; positiveEvidence: ID[]; negativeEvidence: ID[]; missingEvidence: string[]; relationshipKind: string; validityDecision: string };
export type ConstrainedIdentityClusters = { clusters: { clusterId: ID; memberIds: ID[]; acceptedEdges: string[] }[]; blockedMerges: string[]; reviewTriangles: string[] }
export type AttributedCanonicalView = { clusterId: ID; values: { field: string; value: string; sourceId: ID; effectiveFrom: string; effectiveTo: string; authorityRule: string }[]; aliases: string[]; disputedClaims: string[] }
export type IdentityCorrectionPlan = { priorGraphVersion: string; proposedVersion: string; changedEdges: string[]; reverseOperations: string[]; downstreamLinks: string[]; sourceRevocationEffects: string[]; promotionBlocked: string[] }
export type EntityResolutionProposal = { resolutionId: ID; candidates: IdentityPairCandidate[]; judgments: PairIdentityJudgment[]; clusters: ConstrainedIdentityClusters; canonical: AttributedCanonicalView[]; revision: IdentityCorrectionPlan; commitAllowed: boolean }
export type IdentityCorrectionReceipt = { receiptId: ID; resolutionId: ID; priorGraphVersion: string; observedGraphVersion: string; status: "applied" | "partial" | "unknown" | "rolled-back"; appliedEdgeIds: ID[]; reversedOperationIds: ID[]; downstreamAcknowledgements: ID[]; sourceRevocationEvidence: ID[]; evidenceRefs: Ref[] };
export type IdentityCorrectionLedger = { revision: number; currentGraphVersion: string; receipts: IdentityCorrectionReceipt[]; consumedReceiptIds: ID[] };
export type IdentityCorrectionReconciliation = { ledger: IdentityCorrectionLedger; outcome: "applied" | "partial" | "unknown" | "rolled-back" | "held" | "replayed"; affectedConsumers: ID[]; followUp: string[] };
export type IdentityApplyAssessment = { matchesPlan: boolean; safeToAdvance: boolean; reasons: string[]; unresolvedConsumers: ID[] };
