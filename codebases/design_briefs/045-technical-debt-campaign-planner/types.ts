export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type DebtCampaign = RecordMeta & {
  repositoryIds: ID[];
  serviceIds: ID[];
  planningHorizon: string;
  sponsorId: ID;
  capacityBudget: number;
};
export type DebtSymptom = RecordMeta & {
  caseId: ID;
  affectedWorkflow: string;
  observation: string;
  frequencyBasis: string;
  evidenceRefs: Ref[];
};
export type DebtMechanism = RecordMeta & {
  caseId: ID;
  symptomIds: ID[];
  proposedCause: string;
  rivalCause: string;
  validationTask: string;
};
export type DebtWorkPackage = RecordMeta & {
  caseId: ID;
  mechanismId: ID;
  boundedChange: string;
  excludedScope: string;
  completionEvidence: string;
};
export type DebtWave = RecordMeta & {
  caseId: ID;
  packageIds: ID[];
  prerequisiteIds: ID[];
  capacityUse: number;
  rollbackPlan: string;
};
export type DebtOutcome = RecordMeta & {
  caseId: ID;
  packageId: ID;
  beforeMeasure: string;
  afterMeasure: string;
  residualSymptoms: string[];
};
export type DraftDebtSymptom = { values: { caseId?: ID; affectedWorkflow?: string; observation?: string; frequencyBasis?: string; evidenceRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type DebtCampaignRequest = { campaignId: ID; evidence: Evidence[]; priorSymptom: DebtSymptom; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; policyClauses: string[]; baseline: DebtOperationalBaseline; capacityHorizon: string; teamOwners: string[]; runbookEvidence: Evidence[]; alertEvidence: Evidence[] };
export type GroundedDebtSymptom = { symptomId: ID; workflow: string; observation: string; frequencyBasis: string; affectedTeams: string[]; severity: string; evidenceRefs: ID[]; duplicateOf: ID[]; confidence: 'observed' | 'unquantified' };
export type DebtMechanismHypothesis = { mechanismId: ID; symptomIds: ID[]; proposedCause: string; rivalCause: string; disconfirmingEvidenceNeeded: string[]; status: 'supported' | 'provisional' | 'rejected' };
export type DebtImprovementPackage = { packageId: ID; mechanismIds: ID[]; behaviorChange: string; nonGoals: string[]; owner: string; consumerDiscovery: string[]; acceptanceEvidence: string[]; scopeBaseline: string; irreversible: boolean };
export type DebtCampaignWave = { waveId: ID; packageIds: ID[]; prerequisites: ID[]; teamCapacity: string; userVisibleMilestone: string; approvalRequired: boolean };
export type DebtCampaignWaves = { waves: DebtCampaignWave[]; cyclicDependencies: ID[]; capacityGaps: string[] };
export type DebtOperationalBaseline = { observedCost: string; exposureDenominator: string; observationWindow: string; workload: string; sourceRefs: ID[] };
export type WorkaroundRetirementCheck = { workaroundId: ID; activeRunbookRefs: ID[]; activeAlertRefs: ID[]; replacementEvidence: string[]; retirementBlocked: boolean };
export type DebtOutcomeAssessment = { status: 'improved' | 'unchanged' | 'worse' | 'unknown'; normalizedComparison: string; denominatorChanges: string[]; unaddressedMechanisms: ID[]; workaroundBlockers: ID[] };
export type DebtCampaignProposal = { campaignId: ID; symptoms: GroundedDebtSymptom[]; mechanisms: DebtMechanismHypothesis[]; packages: DebtImprovementPackage[]; waves: DebtCampaignWaves; outcome: DebtOutcomeAssessment; approvalNeeded: boolean; commitAllowed: boolean };
export type DebtMeasurementReceipt = { receiptId: ID; campaignId: ID; packageId: ID; baselineMeasure: string; observedMeasure: string; normalizedSignal: "improved" | "unchanged" | "worse" | "unknown"; baselineDenominator: string; observedDenominator: string; baselineWindow: string; observedWindow: string; measureDefinition: string; residualSymptoms: string[]; workaroundStatus: string; evidenceRefs: ID[]; observedAt: DateTime };
export type DebtObservedOutcome = { receiptId: ID; packageId: ID; disposition: "improved" | "unchanged" | "worse" | "unknown"; comparable: boolean; denominatorChanges: string[]; residualSymptoms: string[]; evidenceRefs: ID[]; nextReview: string[] };
export type DebtCampaignLedger = { revision: number; outcomes: DebtObservedOutcome[]; receipts: DebtMeasurementReceipt[]; consumedReceiptIds: ID[] };
export type DebtOutcomeComparison = { comparable: boolean; disposition: "improved" | "unchanged" | "worse" | "unknown"; reasons: string[]; followUp: string[] };
export type DebtOutcomeReconciliation = { ledger: DebtCampaignLedger; outcome: DebtObservedOutcome; receiptDisposition: "recorded" | "replayed" | "held" };
