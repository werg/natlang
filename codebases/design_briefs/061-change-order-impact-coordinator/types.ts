export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type ChangeOrderCase = RecordMeta & {
  projectId: ID;
  instructionIds: ID[];
  baselineScopeId: ID;
  coordinatorId: ID;
  affectedArea: string;
};
export type SiteInstruction = RecordMeta & {
  caseId: ID;
  issuerId: ID;
  requestedChange: string;
  issuedAt: DateTime;
  authorityScope: string;
};
export type ScopeDifference = RecordMeta & {
  caseId: ID;
  instructionId: ID;
  baselineRequirement: string;
  proposedRequirement: string;
  changeKind: string;
};
export type ConstructionImpact = RecordMeta & {
  caseId: ID;
  differenceId: ID;
  affectedTaskIds: ID[];
  interfaceIds: ID[];
  scheduleConsequence: string;
};
export type ChangeOption = RecordMeta & {
  caseId: ID;
  impactIds: ID[];
  proposedWork: string;
  costBasis: string;
  scheduleAlternative: string;
};
export type ChangePackage = RecordMeta & {
  caseId: ID;
  optionId: ID;
  evidenceRefs: Ref[];
  approvalRefs: Ref[];
  residualDispute: string;
};
export type DraftSiteInstruction = { values: { caseId?: ID; issuerId?: ID; requestedChange?: string; issuedAt?: DateTime; authorityScope?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type ChangeOrderDossier = { caseId: ID; instruction: SiteInstruction; writtenConfirmations: InstructionConfirmation[]; fieldReports: WorkEvent[]; drawingRevisions: LocalDrawingRevision[]; schedule: ScheduleActivity[]; costItems: ChangeCostItem[]; dependencies: ScopeDependency[]; evidence: Evidence[]; state?: ChangeOrderState };
export type InstructionConfirmation = { instructionRef: Ref; receivedAt: string; signer: ID; scope: string; authorityRef: Ref | null; disputed: boolean };
export type WorkEvent = { eventId: ID; workPackageId: ID; observedAt: string; description: string; actualCost: string; evidenceRef: Ref };
export type LocalDrawingRevision = { drawingId: ID; revision: string; changedAreas: ID[]; unchangedAreas: ID[]; referenceOnly: boolean; evidenceRef: Ref };
export type ScheduleActivity = { activityId: ID; predecessorIds: ID[]; plannedStart: string; plannedFinish: string; remainingDuration: string; availableFloat: string; affectedByChange: boolean };
export type ChangeCostItem = { costId: ID; workPackageId: ID; changeId: ID; quantity: string; rate: string; evidenceRef: Ref; overlapGroup: ID | null };
export type ScopeDependency = { sourceArea: ID; targetPackage: ID; dependency: string; evidenceRef: Ref };
export type AuthorityFinding = { confirmed: boolean; disputedInstructionRefs: Ref[]; preApprovalEventIds: ID[]; newScopeRefs: Ref[] };
export type ChangeOrderImpactPacket = { caseId: ID; authority: AuthorityFinding; affectedAreas: ID[]; unaffectedAreas: ID[]; scopeBlockers: ID[]; dependencies: ID[]; dependencyBlockers: ID[]; floatAnalysis: string[]; nonDuplicatedCostItems: ID[]; excludedDuplicateCostIds: ID[]; costBlockers: ID[]; proposalStatus: string; state: ChangeOrderState; commitAllowed: boolean };
export type ScopePartition = { affectedAreas: ID[]; unaffectedAreas: ID[]; documentationOnly: boolean; unresolvedAreas: ID[] };
export type ScheduleImpact = { findings: string[]; blockers: string[]; affectedActivityIds: ID[] };
export type CostDeduplication = { acceptedIds: ID[]; duplicateIds: ID[]; unresolvedIds: ID[] };
export type ChangeOrderEdition = { sourceKey: string; version: number; instruction: SiteInstruction; confirmations: InstructionConfirmation[]; fieldReports: WorkEvent[]; drawingRevisions: LocalDrawingRevision[]; schedule: ScheduleActivity[]; costItems: ChangeCostItem[]; dependencies: ScopeDependency[]; evidence: Evidence[]; authority: AuthorityFinding; affectedAreas: ID[]; unaffectedAreas: ID[]; scopeBlockers: ID[]; dependencyIds: ID[]; dependencyBlockers: ID[]; unresolvedDependencyEdges: DependencyEdgeFinding[]; floatAnalysis: string[]; nonDuplicatedCostItems: ID[]; excludedDuplicateCostIds: ID[]; costBlockers: ID[]; proposalStatus: string };
export type ChangeOrderState = { revision: number; currentSourceKey: string | null; editions: ChangeOrderEdition[]; attributedCostIds: ID[]; unresolvedCostIds: ID[] };
export type ConfirmationScopeAudit = { signerAuthorized: boolean; scopeCoversInstruction: boolean; confirmsExactInstruction: boolean; authorityEvidenceCurrent: boolean; preApprovalEventIds: ID[]; disputed: boolean; missingEvidence: string[] };
export type DrawingRevisionFinding = { drawingId: ID; revision: string; affectedAreas: ID[]; preservedAreas: ID[]; documentationOnly: boolean; evidenceRef: Ref; unresolvedAreas: ID[] };
export type DependencyEdgeFinding = { targetPackage: ID; sourceArea: ID; supported: boolean; evidenceRef: Ref; reachesUnchangedScope: boolean; blocker: string };
export type ActivityFloatFinding = { activityId: ID; affectedByChange: boolean; predecessorClosure: ID[]; predecessorGaps: ID[]; floatOutcome: "absorbed" | "consumed" | "delay-risk" | "unknown"; remainingFloat: string; findings: string[] };
export type CostEventMatch = { costId: ID; workPackageId: ID; eventIds: ID[]; overlapGroup: ID | null; quantityRateBasis: string; evidenceMatch: boolean; disposition: "supported" | "duplicate-candidate" | "unresolved"; reasons: string[] };
export type DependencyTrace = { reachablePackageIds: ID[]; traversedEdges: DependencyEdgeFinding[]; unresolvedEdges: DependencyEdgeFinding[] };
