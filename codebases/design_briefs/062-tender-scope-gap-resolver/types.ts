export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type ScopeReview = RecordMeta & {
  projectId: ID;
  bidIds: ID[];
  requiredScopeId: ID;
  reviewerId: ID;
  tenderRound: string;
};
export type BidScope = RecordMeta & {
  caseId: ID;
  bidderId: ID;
  includedFunctions: string[];
  excludedFunctions: string[];
  allowances: string[];
};
export type ScopeCoverage = RecordMeta & {
  caseId: ID;
  requirementId: ID;
  candidateBidIds: ID[];
  coverageKind: 'full'|'partial'|'none'|'uncertain';
};
export type ScopeInterface = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  parties: ID[];
  gap: string;
  duplicateWork: string;
};
export type ScopeClarification = RecordMeta & {
  caseId: ID;
  bidderId: ID;
  question: string;
  requiredCommitment: string;
  responseRef: Ref;
};
export type ResponsibilityMatrix = RecordMeta & {
  caseId: ID;
  coverageIds: ID[];
  interfaceIds: ID[];
  acceptedBidRefs: Ref[];
  residualGaps: string[];
};
export type DraftBidScope = { values: { caseId?: ID; bidderId?: ID; includedFunctions?: string[]; excludedFunctions?: string[]; allowances?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type TenderComparisonDossier = { caseId: ID; bidScopes: BidScope[]; requirements: TenderRequirement[]; bidRevisions: TenderRevision[]; clarifications: TenderClarification[]; interfacePairs: TradeInterface[]; commissioningRequirements: ID[]; evidence: Evidence[]; comparisonState?: TenderComparisonState };
export type TenderRequirement = { requirementId: ID; function: string; tradeId: ID; revision: string; mandatory: boolean; evidenceRef: Ref };
export type TenderRevision = { bidderId: ID; revisionId: ID; receivedAt: string; removedInclusions: ID[]; changedExclusions: ID[]; supersedes: ID | null };
export type TenderClarification = { clarificationId: ID; bidderId: ID; requirementId: ID; response: string; issuedAt: string; authorityRef: Ref; affectsInterfaceIds: ID[] };
export type TradeInterface = { interfaceId: ID; tradeA: ID; tradeB: ID; connection: string; acceptanceOwner: ID | null; acceptanceEvidence: Ref | null };
export type TenderCoverageCell = { bidderId: ID; requirementId: ID; status: string; priceBasis: string; revisionId: ID; evidenceRefs: Ref[] };
export type TenderSeam = { interfaceId: ID; responsibleBidderId: ID | null; acceptanceTask: string; gap: string; supportingCellIds: ID[]; unresolvedBoundary: string };
export type TenderComparison = { cells: TenderCoverageCell[]; seams: TenderSeam[]; uncertainAllowanceIds: ID[]; functionalOmissions: ID[]; invalidatedCellIds: ID[]; rankable: boolean; blockers: string[]; comparisonState: TenderComparisonState };
export type ClassifiedBidCells = { cells: TenderCoverageCell[]; uncertainAllowanceIds: ID[]; blockers: string[]; revisionPins: BidRevisionPin[]; allowanceAudit: AllowanceAssessment[] };
export type ClarificationResult = { cells: TenderCoverageCell[]; uncertainAllowanceIds: ID[]; invalidatedCellIds: ID[]; blockers: string[]; appliedClarificationIds: ID[] };

export type BidRevisionPin = { bidderId: ID; currentRevisionId: ID | null; supersededRevisionIds: ID[]; conflict: string };
export type AllowanceAssessment = { bidderId: ID; requirementId: ID; allowanceId: ID; quantityBasis: string; firmCoverage: boolean; uncertainty: string };
export type TradeBoundaryAssessment = { interfaceId: ID; bidderAssumptions: string[]; ownerCandidateIds: ID[]; unassigned: boolean };
export type ClarificationInvalidation = { invalidatedCellIds: ID[]; affectedClarificationIds: ID[]; staleRevisionIds: ID[] };
export type TenderComparisonRevision = { sourceKey: string; revision: number; tenderCaseId: ID; bidRevisionPins: BidRevisionPin[]; cells: TenderCoverageCell[]; seams: TenderSeam[]; clarificationIds: ID[]; invalidatedCellIds: ID[]; commissioningRequirementIds: ID[]; functionalOmissions: ID[]; blockers: string[]; rankable: boolean };
export type TenderComparisonState = { revision: number; currentSourceKey: string | null; revisions: TenderComparisonRevision[]; cellLineage: TenderCoverageCell[]; supersededRevisionIds: ID[] };
