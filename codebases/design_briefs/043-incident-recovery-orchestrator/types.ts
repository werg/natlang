export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type Incident = RecordMeta & {
  serviceIds: ID[];
  reportedImpact: string;
  commanderId: ID;
  startedAt: DateTime;
  runbookIds: ID[];
};
export type IncidentImpact = RecordMeta & {
  caseId: ID;
  affectedFunctions: string[];
  affectedCohorts: string[];
  observedSymptoms: string[];
  unknownScope: string;
};
export type IncidentHypothesis = RecordMeta & {
  caseId: ID;
  mechanism: string;
  supportingRefs: Ref[];
  contradictingRefs: Ref[];
  discriminatingCheck: string;
};
export type DiagnosticAction = RecordMeta & {
  caseId: ID;
  hypothesisIds: ID[];
  approvedOperationId: ID;
  expectedInformation: string;
  operationalRisk: string;
};
export type RecoveryStep = RecordMeta & {
  caseId: ID;
  operationId: ID;
  prerequisiteIds: ID[];
  approvalRef: Ref;
  rollbackCondition: string;
};
export type RestorationFinding = RecordMeta & {
  caseId: ID;
  function: string;
  verificationRefs: Ref[];
  residualImpact: string;
  monitoringWindow: string;
};
export type DraftIncidentImpact = { values: { caseId?: ID; affectedFunctions?: string[]; affectedCohorts?: string[]; observedSymptoms?: string[]; unknownScope?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type IncidentRecoveryRequest = { incidentId: ID; incidentEvidence: Evidence[]; priorImpact: IncidentImpact; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; policyClauses: string[]; commanderId: ID | null; allowlistedOperations: string[]; currentServiceState: string; observationWindow: string; pendingActions: PendingIncidentAction[] };
export type ObservedIncidentImpact = { affectedFunctions: string[]; affectedCohorts: string[]; symptoms: { statement: string; evidenceRefs: ID[] }[]; unknownScope: string[]; targetOutcomeChecks: string[] };
export type IncidentCauseSet = { hypotheses: { mechanism: string; supportingRefs: ID[]; contradictingRefs: ID[]; discriminatingCheck: string; provisional: boolean }[] };
export type BoundedDiagnosticPlan = { checks: { operationId: ID; hypothesisIds: ID[]; scope: string; resourceBudget: string; expectedInformation: string; approvalRequired: boolean }[]; rejectedOperations: string[]; scopeExpansionBlocked: boolean };
export type ProposedRecoverySequence = { steps: { operationId: ID; runbookVersion: string; preconditions: string[]; freshStateCheck: string; commanderApprovalRef: string; reversible: boolean }[]; blockers: string[] };
export type PendingIncidentAction = { actionId: ID; operationId: ID; status: string; attemptedAt: string; commanderApprovalRef: string };
export type ActionReconciliation = { actionId: ID; actualState: 'executed' | 'not-executed' | 'unknown'; resultEvidence: ID[]; retryAllowed: boolean; holdReason: string };
export type UserOutcomeRestoration = { status: 'restored' | 'degraded' | 'unknown'; verifiedFunctions: string[]; affectedCohortsChecked: string[]; residualImpact: string[]; recurrenceWindow: string; dataRepairRequired: boolean };
export type IncidentRecoveryProposal = { incidentId: ID; impact: ObservedIncidentImpact; hypotheses: IncidentCauseSet; diagnostics: BoundedDiagnosticPlan; recovery: ProposedRecoverySequence; timedOut: ActionReconciliation[]; restoration: UserOutcomeRestoration; open: boolean; commitAllowed: boolean };
export type IncidentActionReceipt = { receiptId: ID; incidentId: ID; actionId: ID; operationId: ID; runbookVersion: string; commanderId: ID; commanderApprovalRef: ID; freshStateCheckRefs: ID[]; outcome: "executed" | "not-executed" | "unknown" | "partial"; postcondition: "verified" | "not-met" | "unknown"; resultEvidenceRefs: ID[]; observedAt: DateTime };
export type IncidentActionState = { actionId: ID; operationId: ID; status: "complete" | "not-executed" | "partial" | "unknown"; postcondition: "verified" | "not-met" | "unknown"; receiptIds: ID[]; blockers: string[] };
export type IncidentRecoveryBook = { revision: number; incidentId: ID; actions: IncidentActionState[]; receipts: IncidentActionReceipt[]; consumedReceiptIds: ID[] };
export type IncidentActionAssessment = { authorized: boolean; currentStepMatched: boolean; canRetry: boolean; reasons: string[] };
export type IncidentActionReconciliation = { book: IncidentRecoveryBook; outcome: "recorded" | "replayed" | "held"; actionStatus: "complete" | "not-executed" | "partial" | "unknown"; retryAllowed: boolean; nextSteps: string[] };
export type TelemetryCoverageBounds = { unobservedFunctionCohorts: string[]; missingChecks: string[]; evidenceRefs: ID[] };
export type HypothesisEvidenceMatrix = { mechanism: string; supportingRefs: ID[]; contradictingRefs: ID[]; ambiguousRefs: ID[] }[];
export type DiagnosticBudgetDecision = { permittedOperationIds: ID[]; blockedOperations: string[]; scopeLimit: string; rateLimit: string; credentialsUnchanged: boolean };
export type CommanderAuthorityCheck = { commanderId: ID | null; current: boolean; exactApprovalRefs: ID[]; blockedReason: string };
export type UnverifiedOutcomeCoverage = { missingCohortChecks: string[]; prematureChecks: string[]; infrastructureOnlyChecks: string[] };
