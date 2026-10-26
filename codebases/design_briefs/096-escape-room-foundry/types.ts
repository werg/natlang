export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type EscapeRoomProject = RecordMeta & WorkRecord & {
  creativeBrief: string;
  venueId: string;
  propIds: string[];
  designerId: string;
  targetDurationMinutes: number;
};
export type EscapeConstraint = RecordMeta & WorkRecord & {
  caseId: string;
  narrativeGoal: string;
  venueConstraint: string;
  accessibilityRequirement: string;
  prohibitedMechanic: string;
};
export type PuzzleNode = RecordMeta & WorkRecord & {
  caseId: string;
  clueInputs: string[];
  solutionAction: string;
  unlocks: string[];
  alternatePath: string;
};
export type PuzzleArtifact = RecordMeta & WorkRecord & {
  caseId: string;
  puzzleId: string;
  artifactText: string;
  intendedInterpretation: string;
  misleadingInterpretations: string[];
};
export type PuzzlePlaytest = RecordMeta & WorkRecord & {
  caseId: string;
  graphId: string;
  actionSequence: string[];
  deadEnd: string;
  hintNeed: string;
};
export type EscapeOperations = RecordMeta & WorkRecord & {
  caseId: string;
  puzzleIds: string[];
  resetTasks: string[];
  hintLadder: string[];
  staffRecoveryRules: string[];
};
export type DraftEscapeConstraint = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    narrativeGoal?: string;
    venueConstraint?: string;
    accessibilityRequirement?: string;
    prohibitedMechanic?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type PuzzleNodeSpec={id:string;clueIds:string[];requires:string[];solves:string[];propIds:string[];mandatory:boolean;};
export type PropState={id:string;present:boolean;working:boolean;sessionId:string;safeMechanism:boolean;};
export type PropResetObservation={propId:string;present:boolean;working:boolean;sessionId:string;};
export type ResetReadback={roomId:string;roomRevision:number;sessionId:string;propObservations:PropResetObservation[];solvedFlagsCleared:boolean;exitAvailable:boolean;observedAt:string;receiptId:string;};
export type EscapeOperationsCheckpoint={roomId:string;revision:number;resetVerified:boolean;nextSessionId?:string;setupReceiptIds?:string[];resetReceiptIds?:string[];openIncidentIds?:string[];};
export type AccessibilityRoute={id:string;declinedSensoryElements:string[];equivalentClueIds:string[];terminalNodeId:string;};
export type EscapeBrief={id:string;goals:string[];accessibilityNeeds:string[];safetyRules:string[];sessionMinutes:number;};
export type EscapeRoomState={id:string;revision:number;brief:EscapeBrief;nodes:PuzzleNodeSpec[];props:PropState[];clues:string[];routes:AccessibilityRoute[];startNodeId:string;goalNodeIds:string[];exitAlwaysAvailable:boolean;};
export type PuzzleGraphState={id:string;solvedNodeIds:string[];availableNodeIds:string[];enabledNodeIds:string[];terminal:boolean;goalComplete:boolean;};
export type PuzzleStructureAudit={roomId:string;revision:number;validNodeIds:string[];duplicateNodeIds:string[];invalidReferences:string[];propCoverageGaps:string[];};
export type PuzzleFrontierAnalysis={states:PuzzleGraphState[];reachableNodeIds:string[];unreachableGoalIds:string[];softLockStates:string[];};
export type PuzzleProof={graphRevision:number;reachableNodeIds:string[];unreachableGoalIds:string[];cycleNodeIds:string[];softLockStates:string[];exitInvariantHolds:boolean;propCoverageGaps:string[];states?:PuzzleGraphState[];invalidReferences?:string[];duplicateNodeIds?:string[];};
export type PuzzleVerification=PuzzleProof & {states:PuzzleGraphState[];invalidReferences:string[];duplicateNodeIds:string[];};
export type PuzzleProofCheck={matches:boolean;differences:string[];};
export type PlaytestFinding={scenario:string;stateId:string;kind:'misread'|'out-of-order'|'lost-prop'|'accessibility'|'repeat-action'|'soft-lock';disposition:'safe-recovery'|'repair'|'block';affectedNodeIds:string[];staffAction:string;};
export type FailureCase={stateId:string;kind:'lost-prop'|'out-of-order'|'accessibility'|'repeat-action';scenario:string;affectedPropId:string;affectedNodeIds:string[];variantId:string;};
export type EscapeOperationsPacket={room:EscapeRoomState;proof:PuzzleProof;playtests:PlaytestFinding[];setup:string[];reset:string[];hints:string[];staffRecovery:string[];checkpoint:EscapeOperationsCheckpoint;status:'blocked'|'review'|'ready-for-safety-and-playtest-review';blockers?:string[];};

export type EscapeResetResult={status:"verified"|"blocked";checkpoint:EscapeOperationsCheckpoint;findings:string[];receiptId?:string;};
export type CycleAssessment={cycleNodeIds:string[];blockedGoals:string[];affectedTerminalStateIds:string[];findings:string[];};
export type ExitAvailabilityReview={verified:boolean;evidenceIds:string[];blockers:string[];};
