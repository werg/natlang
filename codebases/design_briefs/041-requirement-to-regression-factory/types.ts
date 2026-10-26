export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type RegressionProject = RecordMeta & {
  repositoryId: ID;
  requirementIds: ID[];
  targetRevision: string;
  testOwnerId: ID;
  sandboxProfileId: ID;
};
export type BehavioralRequirement = RecordMeta & {
  caseId: ID;
  precondition: string;
  action: string;
  expectedOutcome: string;
  forbiddenOutcome: string;
};
export type TestPartition = RecordMeta & {
  caseId: ID;
  requirementId: ID;
  inputClass: string;
  boundary: string;
  interactionIds: ID[];
};
export type RegressionTest = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  setupSpec: string;
  actionSpec: string;
  assertionSpec: string;
};
export type TestFailureFinding = RecordMeta & {
  caseId: ID;
  testId: ID;
  failureKind: 'product'|'test'|'environment'|'uncertain';
  evidenceRefs: Ref[];
};
export type SuiteQualification = RecordMeta & {
  caseId: ID;
  testIds: ID[];
  mutationIds: ID[];
  survivingMutations: ID[];
  coverageGaps: string[];
};
export type DraftBehavioralRequirement = { values: { caseId?: ID; precondition?: string; action?: string; expectedOutcome?: string; forbiddenOutcome?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type RegressionFactoryRequest = { projectId: ID; requirementNotes: Evidence[]; sourceEvidence: Evidence[]; approvedInterpretations: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; changedRequirementIds: ID[]; currentRequirement: BehavioralRequirement; targetRevision: string; sandboxProfile: string; credentialScope: string; fixtureProvenance: string[]; requiredSeededDefects: string[]; priorRuns: TestRunEvidence[] };
export type BehavioralContract = { requirementId: ID; preconditions: string[]; action: string; expectedOutcome: string; forbiddenOutcome: string[]; evidenceRefs: ID[]; interpretationStatus: 'approved' | 'unresolved' };
export type RegressionPartition = { requirementId: ID; distinction: string; inputCondition: string; stateCondition: string; expectedBehavior: string; failureBehavior: string; evidenceRefs: ID[] };
export type RegressionTestSpec = { id: ID; partitionId: ID; setup: string[]; action: string; assertion: string; cleanup: string[]; oracleEvidenceRefs: ID[]; independentOracle: boolean };
export type SandboxEligibility = { eligible: boolean; blockedReasons: string[]; credentialBoundary: string; fixturePrivacyFindings: string[]; executionPlan: string };
export type TestRunEvidence = { runId: ID; testId: ID; status: string; startedAt: string; endedAt: string; loadContext: string; traceRefs: ID[]; environmentStatus: string };
export type TestFailureClassification = { testId: ID; class: 'product-defect' | 'oracle-defect' | 'fixture-defect' | 'harness-failure' | 'environment-unavailable' | 'flaky'; contractEvidence: ID[]; excludedFromCoverage: boolean; reason: string };
export type FlakyRunEvidence = { preservedRunIds: ID[]; measuredLoadWindows: string[]; retryBudget: string; classifiedFlakes: ID[] };
export type RegressionSuiteQualification = { status: 'qualified' | 'blocked' | 'review'; requiredDefectsDetected: ID[]; undetectedDefects: ID[]; uncoveredRequirements: ID[]; removedAssertions: ID[]; retainedRegressionIds: ID[] };
export type AssertionInvalidationPlan = { invalidatedTestIds: ID[]; preservedTestIds: ID[]; awaitingInterpretation: ID[]; priorSuiteVersion: string };
export type RegressionPlanProposal = { projectId: ID; behavior: BehavioralContract[]; partitions: RegressionPartition[]; tests: RegressionTestSpec[]; sandbox: SandboxEligibility; failures: TestFailureClassification[]; qualification: RegressionSuiteQualification; revisionPlan: AssertionInvalidationPlan; commitAllowed: boolean };
export type RegressionExecutionReceipt = { receiptId: ID; runId: ID; projectId: ID; suiteVersion: string; testId: ID; sourceRevision: string; observedStatus: "passed" | "failed" | "aborted" | "unknown"; environmentStatus: string; traceRefs: ID[]; seed: string; repeatedRunIds: ID[] };
export type RegressionExecutionResult = { testId: ID; runId: ID; classification: 'product-defect' | 'oracle-defect' | 'fixture-defect' | 'harness-failure' | 'environment-unavailable' | 'flaky' | 'pass' | 'unresolved'; coveredRequirementIds: ID[]; defectEvidence: ID[]; explanation: string };
export type RegressionExecutionBook = { revision: number; suiteVersion: string; results: RegressionExecutionResult[]; receipts: RegressionExecutionReceipt[]; consumedReceiptIds: ID[] };
export type RegressionExecutionReview = { book: RegressionExecutionBook; outcome: "recorded" | "replayed" | "held"; suiteQualification: "qualified" | "blocked" | "review"; nextSteps: string[] };
