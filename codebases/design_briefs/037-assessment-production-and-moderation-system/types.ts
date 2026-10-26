export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type AssessmentProject = RecordMeta & {
  competencyIds: ID[];
  audience: string;
  durationMinutes: number;
  assessmentOwnerId: ID;
  stakes: string;
};
export type AssessmentBlueprint = RecordMeta & {
  caseId: ID;
  competencyIds: ID[];
  taskKinds: string[];
  coverageWeights: Ratio[];
  excludedConstructs: string[];
};
export type AssessmentItem = RecordMeta & {
  caseId: ID;
  competencyIds: ID[];
  prompt: string;
  responseFormat: string;
  allowedResources: string[];
};
export type ScoringRubric = RecordMeta & {
  caseId: ID;
  itemId: ID;
  criteria: string[];
  scoreLevels: string[];
  validAlternativeAnswers: string[];
};
export type ItemChallenge = RecordMeta & {
  caseId: ID;
  itemId: ID;
  ambiguousInterpretation: string;
  counterexampleAnswer: string;
  proposedRepair: string;
};
export type ModerationDecision = RecordMeta & {
  caseId: ID;
  responseId: ID;
  originalScore: number;
  reviewedScore: number;
  evidenceReason: string;
};
export type DraftAssessmentBlueprint = { values: { caseId?: ID; competencyIds?: ID[]; taskKinds?: string[]; coverageWeights?: Ratio[]; excludedConstructs?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type CompetencyTrace = { competencyId: ID; evidenceTasks: ID[]; weight: Ratio; excludedConstructs: string[]; accommodationRules: string[]; durationMinutes: number };
export type AssessmentItemDraft = { itemId: ID; prompt: string; competencyIds: ID[]; intendedEvidence: string[]; contextRequired: string[]; allowedResources: string[]; alternateReadings: string[]; exposureStatus: 'clean' | 'leaked' | 'unknown' };
export type RubricSpec = { itemId: ID; criteria: string[]; scoreBounds: number[]; partialCreditBoundary: string[]; validAlternativeMethods: string[]; ambiguityDisposition: string; rubricVersion: string };
export type ChallengeAdjudication = { itemId: ID; independentSolutions: string[]; alternateReadings: string[]; borderlineResponses: string[]; defects: string[]; disposition: 'pass' | 'repair' | 'block' };
export type RegradePlan = { affectedResponseIds: ID[]; oldRubricVersion: string; newRubricVersion: string; proposedScoreChanges: string[]; approvalNeeded: boolean; applied: boolean };
export type AssessmentEditionPlan = { caseId: ID; competencies: CompetencyTrace[]; items: AssessmentItemDraft[]; rubrics: RubricSpec[]; challenges: ChallengeAdjudication[]; exposureRetirements: ExposureRetirement; regrade: RegradePlan[]; releaseStatus: 'blocked' | 'review' | 'ready'; releaseDecision: EditionReleaseDecision; historicalScoringRetained: boolean; questions: Question[]; commitAllowed: boolean };
export type AssessmentBuildRequest = { assessmentId: ID; purpose: string; sourceEvidence: Evidence[]; policyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; revisedEvidenceIds: ID[]; audience: string; currentBlueprint: AssessmentBlueprint };
export type AnswerabilityLedger = { rows: { itemId: ID; missingKnowledge: string[]; materialReadings: string[]; timingDefect: string; answerable: boolean }[] };
export type MethodDisposition = { status: 'accepted' | 'semantic-review' | 'outside-construct'; criterionRefs: string[]; unresolvedProposition: string };
export type IndependentSolutionSet = { routes: { conclusion: string; workedRoute: string[]; unstatedDependencies: string[]; reading: string }[] };
export type BorderlineReplay = { cases: { responseClass: string; criterionTrace: string[]; disagreementLocation: string }[] };
export type LearnerResponse = { item: AssessmentItemDraft; reasoning: string; responseId: ID; rubricVersion: string; awardedScore: number };
export type ModerationContext = { responseIdsUnderVersion: ID[]; independentReviewerId: ID; authorizedCorrection: boolean };
export type AppealFinding = { method: MethodDisposition; routeToIndependentReview: boolean; affectedResponseIds: ID[]; scoreChanges: string[]; evidenceBasis: string[] };
export type ExposureRetirement = { retiredForFutureUse: ID[]; quarantined: ID[]; preservedEditionIds: ID[]; preservedResponseIds: ID[] };
export type RepairRegressionReport = { repeatedChecks: string[]; unresolvedDefects: string[]; clearedDefects: string[] };
export type EditionReleaseDecision = { status: 'blocked' | 'review' | 'ready'; blockingItemIds: ID[]; reasons: string[]; moderatorQuestions: string[]; regradeRequired: boolean };
export type AssessmentReleaseReceipt = { receiptId: ID; assessmentId: ID; editionId: ID; editionVersion: string; planDigest: string; moderatorId: ID; action: "freeze" | "retire"; evidenceRefs: Ref[]; occurredAt: DateTime };
export type AssessmentEditionRecord = { assessmentId: ID; editionId: ID; editionVersion: string; planDigest: string; status: "frozen" | "retired" | "held"; receiptId: ID; evidenceRefs: Ref[] };
export type AssessmentEditionLedger = { revision: number; editions: AssessmentEditionRecord[]; receipts: AssessmentReleaseReceipt[]; consumedReceiptIds: ID[] };
export type AssessmentReleaseAuthority = { assessmentId: ID; moderatorId: ID; authority: "verified" | "invalid" | "unknown"; allowedActions: string[]; evidenceRefs: Ref[] };
export type EditionReleaseReconciliation = { ledger: AssessmentEditionLedger; outcome: "frozen" | "retired" | "held" | "replayed"; blockers: string[]; followUp: string[] };
export type EditionFreezeReadiness = { ready: boolean; blockers: string[] };
export type ScoringCriteriaSet = { criteria: { itemId: ID; criterion: string; competencyId: ID; evidenceTask: string; maximum: number; minimum: number }[] };
export type ValidMethodInventory = { methods: { itemId: ID; route: string; competencyId: ID; evidenceSatisfied: string[]; status: 'accepted' | 'review' }[] };
export type ReviewQueue = { entries: { itemId: ID; proposition: string; rubricVersion: string; automaticScoring: 'allowed' | 'hold' }[] };
