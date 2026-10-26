export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type StorySeries = RecordMeta & WorkRecord & {
  seriesId: string;
  manuscriptIds: string[];
  canonVersion: string;
  editorId: string;
  creativeBrief: string;
};
export type CanonFact = RecordMeta & WorkRecord & {
  caseId: string;
  entityId: string;
  fact: string;
  validStoryTime: string;
  narratorReliability: string;
};
export type StoryObligation = RecordMeta & WorkRecord & {
  caseId: string;
  characterId: string;
  unresolvedSetup: string;
  motivation: string;
  promisedPayoff: string;
};
export type StoryProposalFinding = RecordMeta & WorkRecord & {
  caseId: string;
  proposal: string;
  affectedCanonIds: string[];
  continuityConflict: string;
  creativeOpportunity: string;
};
export type StoryRevision = RecordMeta & WorkRecord & {
  caseId: string;
  affectedSceneIds: string[];
  revisedMaterial: string;
  preservedArcs: string[];
  tradeoff: string;
};
export type StoryEdition = RecordMeta & WorkRecord & {
  caseId: string;
  revisionIds: string[];
  canonDeltaIds: string[];
  remainingObligations: string[];
  editorApprovalRef: string;
};
export type DraftCanonFact = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    entityId?: string;
    fact?: string;
    validStoryTime?: string;
    narratorReliability?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type SerialEdition = { id:string; revision:string; approved:boolean; scenes:SerialScene[]; canon:CanonAssertion[]; knowledge:KnowledgeEvent[]; obligations:ArcObligation[]; };
export type SerialScene = { id:string; episode:number; order:number; characterIds:string[]; assertedFacts:string[]; revealedSecrets:string[]; narration:'reliable'|'unreliable'|'unknown'; text:string; };
export type CanonAssertion = { id:string; proposition:string; kind:'world-fact'|'author-plan'|'character-belief'|'narration'; subjectId:string; sourceSceneId:string; editionId:string; confidence:'approved'|'draft'|'contested'; };
export type KnowledgeEvent = { characterId:string; secretId:string; sceneId:string; editionId:string; learnedAt:number; evidence:string; };
export type KnowledgeLedgerAnalysis = { events:KnowledgeEvent[]; invalidatedEventIds:string[]; unresolvedExposureIds:string[]; };
export type ArcObligation = { id:string; characterId:string; setupSceneId:string; promise:string; state:'open'|'paid'|'deliberate-mystery'|'waived'; rationale:string; };
export type StoryChange = { proposal:SerialScene; authorIntent:string; requestedCanonChanges:string[]; };
export type ContinuityCheckpoint = { editionId:string; revision:string; checkedSceneIds:string[]; knowledge:KnowledgeEvent[]; openObligationIds:string[]; invalidatedSceneIds:string[]; };
export type ContinuityFinding = { sceneId:string; kind:'chronology'|'knowledge'|'world-rule'|'motivation'|'arc'|'narrative'; status:'consistent'|'conflict'|'needs-editor'; evidenceIds:string[]; explanation:string; repairOptions:string[]; };
export type ContinuityDossier = { editionId:string; proposalId:string; findings:ContinuityFinding[]; repairs:StoryRevision[]; canonAmendmentRequired:boolean; affectedSceneIds:string[]; revisedEdition:SerialEdition; checkpoint:ContinuityCheckpoint; status:'review'|'blocked'|'ready'; };

export type BoundStoryContext = { edition: SerialEdition; change: StoryChange; evidence: EvidenceItem[]; seriesId: string; manuscriptIds: string[]; editorId: string; creativeBrief: string; evidenceGaps: string[] };
