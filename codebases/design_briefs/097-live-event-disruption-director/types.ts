export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type LiveEvent = RecordMeta & WorkRecord & {
  eventId: string;
  agendaIds: string[];
  venueId: string;
  producerId: string;
  commitmentIds: string[];
};
export type EventCommitment = RecordMeta & WorkRecord & {
  caseId: string;
  stakeholderId: string;
  promisedOutcome: string;
  timeWindow: string;
  essentialCondition: string;
};
export type EventDisruption = RecordMeta & WorkRecord & {
  caseId: string;
  affectedResourceIds: string[];
  unavailableFunction: string;
  expectedDuration: string;
  uncertainty: string;
};
export type EventRecoveryOption = RecordMeta & WorkRecord & {
  caseId: string;
  affectedAgendaIds: string[];
  alternativeFormat: string;
  preservedPromises: string[];
  changedPromises: string[];
};
export type EventRunPlan = RecordMeta & WorkRecord & {
  caseId: string;
  optionId: string;
  taskIds: string[];
  resourceReservations: string[];
  revisedTiming: string;
};
export type EventChange = RecordMeta & WorkRecord & {
  caseId: string;
  runPlanId: string;
  staffBriefingIds: string[];
  audienceMessageIds: string[];
  acknowledgmentRefs: string[];
};
export type DraftEventCommitment = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    stakeholderId?: string;
    promisedOutcome?: string;
    timeWindow?: string;
    essentialCondition?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type PromiseLedgerItem={id:string;sourceId:string;kind:'public-promise'|'contract'|'accessibility'|'internal';priority:number;outcome:string;dependencies:string[];owner:string;};
export type EventResource={id:string;kind:string;available:boolean;windowStart:number;windowEnd:number;setupMinutes:number;teardownMinutes:number;};
export type ProgramSegment={id:string;start:number;end:number;duration:number;setupBefore:number;teardownAfter:number;resourceIds:string[];ownerRoles:string[];commitmentIds:string[];format:string;accessibility:string[];};
export type EventProductionState={id:string;revision:number;closingMinute:number;commitments:PromiseLedgerItem[];resources:EventResource[];segments:ProgramSegment[];disruptions:Disruption[];staffAcks:StaffAcknowledgement[];publicRevision:string;};
export type Disruption={id:string;capabilityId:string;reportedAt:number;state:'confirmed-lost'|'estimated-lost'|'restored';confidence:string;affectedSegmentIds:string[];sourceId:string;};
export type StaffAcknowledgement={role:string;revision:string;status:'received'|'pending'|'conflict';receivedAt:number;};
export type RunSheetProof={eventRevision:number;doubleBookedResources:string[];brokenDependencies:string[];closingOverrun:boolean;setupConflicts:string[];switchDeadlines:Record<string,number>;};
export type LiveCheckpoint={eventId:string;revision:number;runSheetRevision:string;reservedResourceIds:string[];acknowledgements:StaffAcknowledgement[];executionEvents:string[];emergencyActive:boolean;producerApprovalRevision?:string;};
export type LiveEventEffect={kind:'reserve-event-resources'|'request-role-acknowledgements'|'publish-event-revision';eventId:string;eventRevision:number;runSheetRevision:string;venueId:string;producerId:string;resourceIds:string[];roleUpdates:StaffAcknowledgement[];idempotencyKey:string;};
export type LiveEventEffectPreparation={effects:LiveEventEffect[];checkpoint:LiveCheckpoint;};
export type ChangePacket={event:EventProductionState;options:string[];proof:RunSheetProof|null;messages:string[];decisionDeadlines:string[];checkpoint:LiveCheckpoint;status:'suspended-emergency'|'blocked'|'awaiting-approval'|'ready-to-announce'|'executing';effects?:LiveEventEffect[];blockers?:string[];};
