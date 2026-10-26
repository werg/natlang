export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type GameWorld = RecordMeta & WorkRecord & {
  worldId: string;
  ruleSetId: string;
  sessionIds: string[];
  directorId: string;
  contentBoundaryIds: string[];
};
export type PlayerIntent = RecordMeta & WorkRecord & {
  caseId: string;
  playerId: string;
  intendedAction: string;
  targetIds: string[];
  ambiguity: string;
};
export type CharacterView = RecordMeta & WorkRecord & {
  caseId: string;
  characterId: string;
  knownFactIds: string[];
  beliefs: string[];
  goals: string[];
  obligations: string[];
};
export type CharacterAction = RecordMeta & WorkRecord & {
  caseId: string;
  characterId: string;
  legalActionId: string;
  intent: string;
  dialogue: string;
};
export type WorldConsequence = RecordMeta & WorkRecord & {
  caseId: string;
  actionIds: string[];
  stateEventIds: string[];
  relationshipChanges: string[];
  unresolvedHooks: string[];
};
export type WorldContinuity = RecordMeta & WorkRecord & {
  caseId: string;
  checkpointId: string;
  unresolvedConflictIds: string[];
  proposedEvent: string;
  affectedEntityIds: string[];
};
export type DraftPlayerIntent = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    playerId?: string;
    intendedAction?: string;
    targetIds?: string[];
    ambiguity?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type PlayerUtterance={playerId:string;characterId:string;text:string;visibleLocation:string;clientActionId:string;};
export type PersistentWorldState={id:string;revision:number;ruleVersion:string;locations:string[];resources:WorldResource[];characters:GameCharacter[];events:WorldEvent[];boundaries:ContentBoundary[];};
export type WorldResource={id:string;ownerCharacterId:string;location:string;quantity:string;revision:number;};
export type GameCharacter={id:string;goals:string[];relationships:string[];knownFactIds:string[];beliefs:CharacterBelief[];memoryEventIds:string[];};
export type CharacterBelief={factId:string;belief:string;sourceEventId:string;certainty:string;};
export type WorldEvent={id:string;revision:number;kind:string;actorId:string;location:string;facts:string[];privateTo:string[];sourceActionId:string;};
export type ContentBoundary={participantId:string;blockedTopics:string[];allowedIntensity:string;revision:string;};
export type LegalAction={id:string;actorId:string;kind:string;targetId:string;location:string;resourceDelta:string;requires:string[];clientActionId:string;};
export type WorldTransitionReceipt={clientActionId:string;worldId:string;priorRevision:number;newRevision:number;outcome:'committed'|'rejected';eventIds:string[];resourceLedgerHash:string;rejectionReason:string;};
export type ActionCheck={actionId:string;eligible:boolean;blockers:string[];evidenceIds:string[];};
export type ActionSelection={actions:LegalAction[];checks:ActionCheck[];questions:string[];blockers:string[];};
export type WorldSimulatorReadback={worldId:string;revision:number;ruleVersion:string;resourceLedgerHash:string;eventIds:string[];receiptedActionIds:string[];world:PersistentWorldState;};
export type ReceiptBindingFacts={receipts:WorldTransitionReceipt[];blockers:string[];actions:{clientActionId:string;receipt:WorldTransitionReceipt|null;conflictingDuplicate:boolean;worldMatches:boolean|null;priorRevisionMatches:boolean|null;actionBound:boolean|null;rejectionReasonPresent:boolean|null;revisionAdvances:boolean|null;readbackWorldMatches:boolean|null;readbackRevisionMatches:boolean|null;readbackRuleVersionMatches:boolean|null;resourceHashMatches:boolean|null;actionReported:boolean|null;missingReceiptEventIds:string[];invalidEventBindings:string[]}[];};
export type ReceiptEventReview={acceptedEvents:WorldEvent[];unresolvedEventIds:string[];blockers:string[];};
export type WorldActionEffectPreparation={effects:WorldActionEffect[];checkpoint:GameCheckpoint;blockers:string[];};
export type WorldActionEffect={kind:'apply-player-actions'|'read-action-status';worldId:string;priorRevision:number;ruleVersion:string;clientActionId:string;actions:LegalAction[];idempotencyKey:string;};
export type PendingWorldAction={clientActionId:string;worldId:string;priorRevision:number;ruleVersion:string;actions:LegalAction[];};
export type WorldReconciliation={receipts:WorldTransitionReceipt[];pendingActionIds:string[];unknownReceiptIds:string[];committedEvents:WorldEvent[];updatedWorld:PersistentWorldState|null;status:'awaiting-simulator'|'reconciled'|'blocked';blockers:string[];};
export type GameCheckpoint={worldId:string;revision:number;ruleVersion:string;pendingActionIds:string[];unknownReceiptIds:string[];pendingActions?:PendingWorldAction[];boundaryRevisions:string[];replayCursor:string;worldSnapshot?:PersistentWorldState;receiptLedger?:WorldTransitionReceipt[];};
export type GameSessionResult={intent:LegalAction[];questions:string[];views:CharacterPerspective[];actions:LegalAction[];receipts:WorldTransitionReceipt[];beliefUpdates:CharacterBelief[];continuityNotes:string[];checkpoint:GameCheckpoint;status:'needs-clarification'|'awaiting-simulator'|'reconciled'|'blocked';effects?:WorldActionEffect[];worldState?:PersistentWorldState;};
export type CharacterPerspective={characterId:string;visibleFacts:string[];permittedMemories:string[];privateFactsExcluded:string[];availableActions:string[];};
export type CheckpointReconciliation={checkpoint:GameCheckpoint;ready:boolean;blockers:string[];};
