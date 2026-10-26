export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type HarvestCoordinator = { id: string };
export type LotRevision = { lotId: string; priorEstimate: string; revisedEstimate: string; sourceVersion: string; exactVariety: string; qualityState: string; readinessWindow: string; uncertain: boolean };
export type HarvestChange = { changedLotIds: string[]; observedEvidence: Evidence[]; rejectedSubstitutionIds: string[]; physicalReceipts: MovementReceipt[]; resourceQuantityLimits?: QuantityLimitCheck[] };
export type MovementReceipt = { id: string; lotId: string; phase: 'harvested' | 'stored' | 'delivered'; quantity: Quantity; observedAt: string; sourceVersion: string };
export type MovementReceiptSum = { totals: Record<string, Quantity>; duplicateReceiptIds: string[]; invalidReceiptIds: string[]; invalidGroups: string[] };
export type QuantityLimitCheck = { limitId: string; limitKind: 'lot-supply'|'commitment-demand'|'resource-capacity'; available: Quantity; assignments: { id: string; quantity: Quantity }[]; evidenceRefs: string[] };
export type ExactQuantityCheck = { limitId: string; limitKind: 'lot-supply'|'commitment-demand'|'resource-capacity'; total: Quantity; available: Quantity; valid: boolean; issue: string; evidenceRefs: string[] };
export type HarvestDeskState = { round: HarvestRound; checkpoint?: HarvestCheckpoint; priorLots: LotRevision[]; readiness: HarvestReadiness[]; commitments: HarvestCommitment[]; matches: HarvestMatch[]; plans: HarvestPlan[]; movementReceipts: MovementReceipt[]; changedSourceIds: string[]; revision: string };
export type FieldDelta = { lots: LotRevision[]; evidence: Evidence[]; unresolved: string[]; affectedCommitments: string[]; resourceQuantityLimits?: QuantityLimitCheck[] };
export type LotSuitability = { lotId: string; commitmentId: string; eligible: boolean; usableQuantity: Quantity; proposedQuantity?: Quantity; reasons: string[]; evidenceIds: string[] };
export type LotAllocation = { edges: LotSuitability[]; allocatedByLot: Record<string, Quantity>; unmetCommitments: string[]; blockedEdges: LotSuitability[]; exactQuantityChecks?: ExactQuantityCheck[] };
export type CapacityFit = { feasible: boolean; plan: HarvestPlan[]; resourceHolds: string[]; overlaps: string[]; quantityConserved: boolean; blockers: string[]; exactCapacityChecks?: ExactQuantityCheck[] };
export type MovementReconciliation = { harvested: Record<string, Quantity>; stored: Record<string, Quantity>; delivered: Record<string, Quantity>; unexplained: string[]; fulfilledCommitments: string[]; };
export type HarvestCheckpoint = { priorRevision: string; revision: string; priorLots: LotRevision[]; readiness: HarvestReadiness[]; commitments: HarvestCommitment[]; matches: HarvestMatch[]; plans: HarvestPlan[]; movementReceipts: MovementReceipt[]; unresolved: string[]; };
export type HarvestControlProposal = { application: 'Agricultural Cooperative Harvest Coordinator'; status: 'ready-for-review' | 'blocked'; changedLots: LotRevision[]; readiness: HarvestReadiness[]; commitments: HarvestCommitment[]; allocation: LotAllocation; resources: CapacityFit; movement: MovementReconciliation; affectedCommitmentIds: string[]; actionBoundary: string; mustPreserve: string[]; blockers: string[]; resumeState: HarvestCheckpoint; };
