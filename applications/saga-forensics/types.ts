/** Immutable fictional evidence; policies returned by this application are data. */
export type InvestigationRequest = {
  caseId: string; targetOrders: string[]; asOfGlobalTime: number;
  objective: string; requiredPostconditions: string[];
};
export type Witness = { sourceId: string; path: string; quote: string };
export type SourceDocument = { id: string; path: string; roleHint: string; text: string };
export type SourceManifest = {
  caseId: string; documents: { id: string; path: string; roleHint: string }[];
  source_scope: string; output_contract: Record<string, string>;
};
export type EvidenceAtom = {
  id: string;
  kind: 'contract' | 'attempt' | 'receipt' | 'clock' | 'handover' | 'cancellation' | 'pool' | 'query' | 'boundary';
  subjectIds: string[]; statement: string; witnesses: Witness[];
};
export type EvidenceIndex = {
  manifest: SourceManifest; documents: SourceDocument[]; atoms: EvidenceAtom[];
  coverage: { sourceId: string; atomIds: string[] }[]; ambiguities: string[];
};
export type ClockRelation = {
  clock: string; globalOffset: number | null; applicableScope: string;
  resetChangesOffset: boolean | null; witnesses: Witness[];
};
export type EventTime = {
  eventId: string; localTime: number | null; clock: string | null;
  globalTime: number | null; kind: string; witnesses: Witness[];
};
export type Timeline = { clocks: ClockRelation[]; events: EventTime[]; unresolved: string[] };
export type Attempt = {
  attemptId: string; orderId: string; system: 'warehouse' | 'bank';
  kind: 'reservation' | 'rental' | 'deposit' | 'refund';
  tenant: string | null; warehouse: string | null; epoch: string | null; key: string;
  quantity: number; unit: 'kits' | 'money'; meaning: string;
  referencedEffectId: string | null; causeId: string | null; witnesses: Witness[];
};
export type PayloadRelation = {
  sameMeaning: boolean | null; differingFields: string[]; explanation: string; witnesses: Witness[];
};
export type IdentityDecision = {
  leftAttemptId: string; rightAttemptId: string;
  relation: 'duplicate' | 'independent' | 'incompatible' | 'unresolved';
  scopeFields: string[]; reason: string; witnesses: Witness[];
};
export type IdentityPartition = {
  attempts: Attempt[]; groups: string[][]; pairDecisions: IdentityDecision[];
  rejectedAttemptIds: string[]; unresolved: string[];
};
export type CausalEdge = {
  beforeId: string; afterId: string;
  kind: 'execution' | 'retry' | 'compensation' | 'custody' | 'policy-boundary';
  witnesses: Witness[];
};
export type Correlation = { identity: IdentityPartition; timeline: Timeline; edges: CausalEdge[]; conflicts: string[] };
export type EffectCandidate = {
  ledgerId: string; effectId: string | null; attemptIds: string[];
  relevantAtomIds: string[];
};
export type EvidenceAssessment = {
  candidate: EffectCandidate;
  observations: { atomId: string; strength: 'commit' | 'replay' | 'intent' | 'acceptance' | 'rejection' | 'bounded-unknown'; witnesses: Witness[] }[];
  possibleStatuses: ('committed' | 'absent')[];
  conflicts: string[];
};
export type Effect = {
  ledgerId: string; effectId: string | null; orderId: string;
  kind: 'reservation' | 'rental' | 'deposit' | 'refund';
  quantity: number; unit: 'kits' | 'money';
  tenant: string | null; key: string | null; referencedEffectId: string | null;
  causeId: string | null; attemptIds: string[];
  possibleStatuses: ('committed' | 'absent')[]; witnesses: Witness[];
};
export type LedgerState = {
  pending: EffectCandidate[]; effects: Effect[]; discardedAttemptIds: string[];
  conflicts: string[]; index: EvidenceIndex; correlation: Correlation;
};
export type EffectLedger = { effects: Effect[]; discardedAttemptIds: string[]; conflicts: string[] };
export type Obligation = {
  obligationId: string; orderId: string; cancellationId: string;
  reservationEffectId: string; paymentEffectId: string;
  releaseQuantity: number; totalRefundEntitlement: number; confirmedRefund: number;
  uncertainRefundLedgerIds: string[]; protectedEffectIds: string[];
  witnesses: Witness[];
};
export type ObligationGraph = {
  obligations: Obligation[]; protectedEffectIds: string[];
  initialStock: number; currentFree: number; inUse: number; currentReserved: number;
  confirmedCash: number; uncertainRefundLedgerIds: string[];
  unresolved: string[]; witnesses: Witness[];
};
export type QueryAnswer = {
  answer: string; possibleStatuses: ('committed' | 'absent')[];
  establishesFinalAbsence: boolean; witnesses: Witness[];
};
export type ObservationChoice = {
  queryId: string; ledgerId: string; cost: number; coverageThrough: number;
  processingBoundary: number; answers: QueryAnswer[]; sufficient: boolean;
  reason: string; witnesses: Witness[];
};
export type ObservationPlan = { selected: ObservationChoice[]; rejected: ObservationChoice[]; unresolved: string[] };
export type RepairAction = {
  actionId: string; kind: 'release' | 'refund'; orderId: string;
  subjectEffectId: string; quantity: number; unit: 'kits' | 'money';
  tenant: string | null; key: string | null; causeId: string;
  dependsOn: string[]; witnesses: Witness[];
};
export type PolicyBranch = {
  conditions: { queryId: string; answer: string; ledgerId: string; status: 'committed' | 'absent' }[];
  actions: RepairAction[];
};
export type PolicyCandidate = {
  coveredObligationIds: string[]; commonActions: RepairAction[]; branches: PolicyBranch[];
  worstCaseWrites: number; observationCost: number; rejectedReasons: string[];
};
export type ActionAssessment = {
  permitted: boolean; necessaryQuantity: number; violatedRules: string[];
  witnesses: Witness[];
};
export type SearchContext = {
  request: InvestigationRequest; index: EvidenceIndex; correlation: Correlation;
  ledger: EffectLedger; obligations: ObligationGraph; observations: ObservationPlan;
};
export type RepairSearchState = {
  context: SearchContext; remainingObligationIds: string[];
  candidates: PolicyCandidate[]; rejected: { obligationId: string; reason: string; witnesses: Witness[] }[];
};
export type RepairPolicy = {
  observations: ObservationPlan; candidate: PolicyCandidate | null;
  obligations: ObligationGraph; blockedReasons: string[];
};
/** protectedEffectIds names whole effects preserved from compensation. Partially consumed reservations
 * are represented by the inUse quantity, not added to this whole-effect protection set. */
export type BranchAccounting = {
  conditions: { queryId: string; answer: string }[];
  startingCash: number; finalCash: number; free: number; inUse: number; reserved: number;
  writeCount: number; protectedEffectIds: string[];
};
export type AuditFinding = { severity: 'defect' | 'unresolved' | 'defended'; proposition: string; witnesses: Witness[] };
export type ForensicReport = {
  caseId: string; identity: IdentityPartition; timeline: Timeline; causalEdges: CausalEdge[];
  ledger: EffectLedger; repair: RepairPolicy; currentCashPossibilities: number[];
  branchAccounting: BranchAccounting[]; findings: AuditFinding[];
  sourceCoverage: { sourceId: string; atomIds: string[] }[];
};
