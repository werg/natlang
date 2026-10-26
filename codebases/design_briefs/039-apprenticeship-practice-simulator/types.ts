export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type PracticeSession = RecordMeta & {
  learnerId: ID;
  scenarioFamily: string;
  procedureId: ID;
  instructorId: ID;
  difficulty: string;
};
export type PracticeScenario = RecordMeta & {
  caseId: ID;
  initialFacts: string[];
  hiddenFacts: string[];
  actorGoals: string[];
  allowedActions: string[];
};
export type LearnerAction = RecordMeta & {
  caseId: ID;
  utterance: string;
  intendedAction: string;
  targetId: ID;
  ambiguity: string;
};
export type PracticeTransition = RecordMeta & {
  caseId: ID;
  actionId: ID;
  stateDelta: string;
  revealedFacts: string[];
  consequence: string;
};
export type PracticeComplication = RecordMeta & {
  caseId: ID;
  trigger: string;
  newFact: string;
  affectedObligationIds: ID[];
  recoveryPaths: string[];
};
export type PracticeFeedback = RecordMeta & {
  caseId: ID;
  actionIds: ID[];
  demonstratedSkills: string[];
  missedEvidence: string[];
  nextPractice: string;
};
export type DraftPracticeScenario = { values: { caseId?: ID; initialFacts?: string[]; hiddenFacts?: string[]; actorGoals?: string[]; allowedActions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type PracticeCheckpoint = { sessionId: ID; version: number; appliedTransitionIds: ID[]; seenChallengeIds: ID[]; lastTransitionId: ID; worldStateJson: string; actorKnowledgeJson: string; turnReceipts?: ApprenticeshipTurnReceipt[] };
export type TrainingWorld = { resources: { id: ID; quantity: number; status: string }[]; visibleFacts: string[]; hiddenFacts: string[]; stateJson: string };
export type ProcedureStep = { id: ID; requiredActions: string[]; preconditions: string[]; unsafeConditions: string[]; recoverySteps: string[] };
export type ObservedAction = { id: ID; utterance: string; actionKind: string; target: string; outcome: string; factRefs: string[] };
export type SimulatedActor = { id: ID; visibleFactIds: string[]; authorizedReferences: string[] };
export type ApprenticeshipTurn = { sessionId: ID; turnId: ID; sourceScenario: PracticeScenario; utterance: string; sourceEvidence: Evidence[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; actor: SimulatedActor; world: TrainingWorld; procedure: ProcedureStep; scenarioFamily: string; learningObjective: string; seed: string; checkpoint: PracticeCheckpoint; actionHistory: ObservedAction[] };
export type UtteranceKind = { kind: 'question' | 'action-proposal' | 'ambiguous'; intent: string; target: string; };
export type ActorAnswer = { answer: string; factRefs: string[]; withheld: string[] };
export type InterpretedUtterance = { kind: UtteranceKind; answer?: ActorAnswer; clarificationNeeded: string[] };
export type TypedTrainingAction = { turnId: ID; kind: string; target: string; preconditions: string[]; policyRef: string; status: 'simulated' | 'clarification' | 'refusal'; clarificationNeeded: string[] };
export type PracticeWorldTransition = { transitionId: ID; actionId: ID; priorStateJson: string; nextStateJson: string; revealedFactIds: string[]; consequence: string; invariantFailures: string[] };
export type ApprenticeshipTurnBinding = { sessionId: ID; turnId: ID; sourceScenario: PracticeScenario; utterance: string; sourceEvidence: Evidence[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; actor: SimulatedActor; world: TrainingWorld; procedure: ProcedureStep; scenarioFamily: string; learningObjective: string; seed: string; actionHistory: ObservedAction[] };
export type ApprenticeshipTurnReceipt = { binding: ApprenticeshipTurnBinding; utterance: InterpretedUtterance; typedAction: TypedTrainingAction; transition: PracticeWorldTransition; challenge: SeededPracticeChallenge; debrief: ProcedureDebrief };
export type TurnReplayCheck = { status: 'new' | 'replay' | 'conflict' | 'unresolved'; receipt: ApprenticeshipTurnReceipt | null; blockers: string[] };
export type SeededPracticeChallenge = { id: ID; seed: string; trigger: string; visibleFactSnapshot: string[]; recoveryPaths: string[]; deferredReason: string };
export type ProcedureDebrief = { plannedActionKind: string; plannedTarget: string; simulatedActionId: ID; simulatedTransitionId: ID; simulatedConsequence: string; actionEvidence: string[]; currentActionObserved: boolean; missedObligations: string[]; effectiveChecks: string[]; outcomeLuck: string; nextPracticeTarget: string };
export type ApprenticeshipTurnResult =
  | { sessionId: ID; status: 'question-answered'; utterance: InterpretedUtterance; typedAction: null; transition: null; challenge: null; debrief: null; blockers: string[]; checkpoint: PracticeCheckpoint; replayed: boolean; commitAllowed: boolean }
  | { sessionId: ID; status: 'clarification-needed'; utterance: InterpretedUtterance; typedAction: TypedTrainingAction | null; transition: null; challenge: null; debrief: null; blockers: string[]; checkpoint: PracticeCheckpoint; replayed: boolean; commitAllowed: boolean }
  | { sessionId: ID; status: 'simulated'; utterance: InterpretedUtterance; typedAction: TypedTrainingAction; transition: PracticeWorldTransition; challenge: SeededPracticeChallenge; debrief: ProcedureDebrief; blockers: string[]; checkpoint: PracticeCheckpoint; replayed: boolean; commitAllowed: boolean }
  | { sessionId: ID; status: 'replay-conflict' | 'replay-unresolved'; utterance: null; typedAction: null; transition: null; challenge: null; debrief: null; blockers: string[]; checkpoint: PracticeCheckpoint; replayed: boolean; commitAllowed: boolean };
export type InvariantCheck = { passed: boolean; target: string; violatedBound: string; evidence: string[] };
export type ChallengeCandidate = { id: ID; seed: string; trigger: string; expectedDecision: string; allowedRecovery: string[]; };
export type ChallengeFairnessCheck = { fair: boolean; recoverableSequence: string[]; hiddenPrerequisites: string[]; retroactiveContradictions: string[] };
