export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; workflowStatus: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type NegotiationExercise = RecordMeta & WorkRecord & {
  learningObjectives: string[];
  participantIds: string[];
  scenarioFamily: string;
  instructorId: string;
  conductPolicyId: string;
};
export type NegotiationScenario = RecordMeta & WorkRecord & {
  caseId: string;
  issues: string[];
  roleIds: string[];
  hiddenPreferences: string[];
  feasibleAgreementRules: string[];
};
export type NegotiationRole = RecordMeta & WorkRecord & {
  caseId: string;
  participantId: string;
  privateFacts: string[];
  authorityLimits: string[];
  permittedTactics: string[];
};
export type TrainingOffer = RecordMeta & WorkRecord & {
  caseId: string;
  proposerId: string;
  terms: string[];
  conditions: string[];
  status: 'proposal'|'counteroffer'|'accepted'|'withdrawn';
};
export type CounterpartAction = RecordMeta & WorkRecord & {
  caseId: string;
  roleId: string;
  legalActionId: string;
  response: string;
  privateObjectiveBasis: string;
};
export type NegotiationDebrief = RecordMeta & WorkRecord & {
  caseId: string;
  agreementIds: string[];
  observedSkills: string[];
  missedOpportunities: string[];
  nextExercise: string;
};
export type DraftNegotiationScenario = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    issues?: string[];
    roleIds?: string[];
    hiddenPreferences?: string[];
    feasibleAgreementRules?: string[];
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type ArenaIssue={id:string;name:string;unit:string;minimum:string;maximum:string;};
export type ArenaRole={id:string;authority:string[];privatePriorities:string[];reservationConditions:string[];conductLimits:string[];disclosableFacts:string[];};
export type ArenaScenario={id:string;revision:number;issues:ArenaIssue[];roles:ArenaRole[];learningGoals:string[];allowedTactics:string[];prohibitedTactics:string[];debriefPolicy:string[];acceptanceRule:string;};
export type ParsedArenaOffer={turnId:string;fromRole:string;terms:Record<string,string>;conditions:string[];expiresAt:string;commitment:'exploratory'|'firm'|'conditional-acceptance'|'withdrawal';clarifications:string[];};
export type FeasibilityArtifact={scenarioRevision:number;offerId:string;feasible:boolean;authorityValid:boolean;withinIssueBounds:boolean;utilityBandByRole:Record<string,string>;hiddenToLearner:string[];verifier:string;};
export type CounterpartChoice={roleId:string;actionId:string;terms:Record<string,string>;language:string;revealedFacts:string[];conductCheck:string;};
export type ArenaTurn={turnId:string;speaker:string;utterance:string;offer:ParsedArenaOffer|null;counterpart:CounterpartChoice|null;state:'open'|'clarification-needed'|'accepted'|'withdrawn'|'rejected';};
export type ArenaCheckpoint={scenarioId:string;revision:number;turns:ArenaTurn[];openOfferIds:string[];acceptedTerms:string[];difficultyRevision:string;debriefReleaseScope:string;};
export type ArenaRunResult={scenario:ArenaScenario;turns:ArenaTurn[];clarificationQuestions:string[];feasibility:FeasibilityArtifact[];privateFeedback:string[];debrief:string[];checkpoint:ArenaCheckpoint;status:'in-progress'|'clarification-needed'|'agreement'|'withdrawn'|'debrief-ready';};
