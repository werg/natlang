export type Decimal = string;
export type Ratio = { numerator: string; denominator: string; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };
export type StageAudit = { verdict: 'pass' | 'fail' | 'needs-review'; gate: string; findings: string[]; repairTargets: string[]; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type TheatreConstraint = { caseId: string; sceneId: string; castRequirements: string[]; propRequirements: string[]; consentBoundary: string; };
export type SceneVariant = { caseId: string; sceneId: string; variantText: string; preservedBeat: string; changedDependencies: string[]; };
export type TheatreCuePlan = { caseId: string; variantIds: string[]; cueOrder: string[]; responsibleCastIds: string[]; timingWindow: string; };
export type TheatreRecovery = { caseId: string; disruptedSceneIds: string[]; recoveryMaterial: string; preservedFacts: string[]; approvalRef: string; };
export type ImmersiveProduction = RecordMeta & { kind: 'ImmersiveProduction'; subject: string; productionId: string; sceneIds: string[]; castIds: string[]; venueId: string; stageManagerId: string; fields: Record<string,string>; };
export type TheatreCheckpoint = { id: string; productionId: string; priorCheckpointId: string; version: string; state: 'approved' | 'proposed' | 'held' | 'unknown'; sceneOrder: string[]; executedCueReceiptIds: string[]; audienceKnowledgeByGroup: Record<string,string[]>; characterKnowledgeById: Record<string,string[]>; pendingCueIds: string[]; reservedResourceWindows: ResourceWindow[]; safetyHold: boolean; holdSignalIds: string[]; changedSourceIds: string[]; };
export type LiveShowState = { production: ImmersiveProduction; checkpoint: TheatreCheckpoint; scriptEvidence: EvidenceItem[]; productionEvidence: EvidenceItem[]; safetyRules: EvidenceItem[]; };
export type TheatreEvent = { request: string; evidence: EvidenceItem[]; updates: EvidenceItem[]; sceneIds: string[]; };
export type StageManager = { id: string; };
export type DraftTheatreScene = { values: { id?: string; caseId?: string; sourceRefs?: string[]; state?: 'draft' | 'qualified' | 'blocked'; kind?: string; subject?: string; evidenceIds?: string[]; fields?: Record<string,string>; status?: string; prerequisiteFacts?: string[]; characterKnowledge?: string[]; intendedBeat?: string; branchConditions?: string[]; revealEdges?: string[]; knowledgeByCharacter?: Record<string,string[]>; knowledgeByAudienceGroup?: Record<string,string[]>; }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type IncidentKind = 'participant-decline' | 'cast-absence' | 'prop-failure' | 'knowledge-divergence' | 'safety-stop' | 'ordinary';
export type ShowIncident = { kind: IncidentKind; evidenceIds: string[]; affectedScenes: string[]; affectedPeople: string[]; affectedProps: string[]; stopSignalId: string; description: string; unresolved: string[]; };
export type RevealEdge = { fromScene: string; toScene: string; fact: string; audienceGroups: string[]; characters: string[]; sourceIds: string[]; status: 'supported' | 'disputed' | 'missing'; };
export type SceneRevealGraph = { productionId: string; sceneOrder: string[]; edges: RevealEdge[]; blockedRoutes: string[]; gaps: string[]; sourceIds: string[]; };
export type KnowledgeLedger = { byCharacter: Record<string,string[]>; byAudienceGroup: Record<string,string[]>; protectedFacts: string[]; unprovenTransfers: string[]; };
export type TheatreScene = { id: string; caseId: string; sourceRefs: string[]; state: 'qualified' | 'blocked'; sceneId: string; prerequisiteFacts: string[]; characterKnowledge: string[]; audienceKnowledge: Record<string,string[]>; intendedBeat: string; branchConditions: string[]; revealEdges: RevealEdge[]; };
export type DramaturgyMap = { graph: SceneRevealGraph; knowledge: KnowledgeLedger; scenes: TheatreScene[]; unresolved: SceneCandidate[]; affectedScenes: string[]; questions: string[]; ready: boolean; };
export type ResourceWindow = { resourceId: string; resourceKind: 'cast' | 'venue' | 'prop' | 'accessibility' | 'stage-control'; sceneId: string; window: string; startsAt?: string; endsAt?: string; exclusivity?: 'exclusive' | 'shared' | 'unknown'; status: 'confirmed' | 'reserved' | 'unavailable' | 'unverified'; evidenceIds: string[]; };
export type ConsentRoute = { sceneId: string; participantId: string; chosenPath: string; participationRequired: boolean; equivalentOutcome: boolean; evidenceIds: string[]; };
export type ResourceTimeline = { windows: ResourceWindow[]; conflicts: string[]; consentRoutes: ConsentRoute[]; unavailable: string[]; unresolved: string[]; } ;
export type VariantCandidate = { id: string; sceneId: string; route: string; preservedBeat: string; requiredCast: string[]; requiredProps: string[]; knowledgeDelta: string[]; participation: string; approvalStatus: 'rehearsed-approved' | 'rewrite-review-required' | 'unknown'; evidenceIds: string[]; };
export type RouteComparison = { sceneId: string; candidates: VariantCandidate[]; selectedId: string; reasons: string[]; blockers: string[]; };
export type CueNode = { id: string; sceneId: string; ownerId: string; resourceIds: string[]; after: string[]; window: string; safetyCritical: boolean; state: 'proposed' | 'awaiting-acknowledgment'; sourceIds: string[]; };
export type CueSequence = { order: string[]; cues: CueNode[]; conflicts: string[]; pendingAcknowledgments: string[]; executionReceipts: string[]; ready: boolean; };
export type ContinuityReplay = { sourceCheckpoint: string; newCheckpointId: string; actualPath: string[]; unobservedPlans: string[]; knowledgeByAudienceGroup: Record<string,string[]>; safetyHold: boolean; holdSignalIds: string[]; resetTasks: string[]; blockers: string[]; proposedCheckpoint: TheatreCheckpoint; };
export type TheatreRunView = { affectedScenes: string[]; selectedRoutes: string[]; resourceChanges: string[]; cueOwnersAwaitingAcknowledgment: string[]; preservedFacts: string[]; blockers: string[]; recipient: string; };
export type TheatreShowControl = { safetyHold: boolean; holdSignalIds: string[]; checkpointId: string; executedReceiptIds: string[]; pendingAcknowledgments: string[]; };
export type TheatreControlProposal = { application: 'Immersive Theatre Stage Manager'; status: 'ready-for-review' | 'blocked'; kind: 'safety-hold' | 'performance-change'; incident: ShowIncident; dramaturgy?: DramaturgyMap; resources?: ResourceTimeline; routeComparisons?: RouteComparison[]; cues?: CueSequence; continuity: ContinuityReplay; view: TheatreRunView; showControl: TheatreShowControl; questions: string[]; coverage: string[]; checks: string[]; actionBoundary: string; mustPreserve: string; resumeState: TheatreCheckpoint; };
export type SceneCandidate = { production: ImmersiveProduction; sceneId: string; evidence: EvidenceSlice; knowledge: KnowledgeLedger; graph: SceneRevealGraph; };

export type StopProcedureMatch = { procedureId: string; safetyLeadId: string; scope: string[]; current: boolean; blockers: string[]; evidenceIds: string[]; };
export type FrozenCueSet = { frozenCueIds: string[]; alreadyExecutedIds: string[]; heldDependencies: string[]; };

export type ResourceConflicts = { conflicts: string[]; unavailable: string[]; unverified: string[]; windows: ResourceWindow[]; };
export type ConsentAssessment = { routes: ConsentRoute[]; blockers: string[]; affectedPeople: string[]; };
export type VariantSet = { comparisons: RouteComparison[]; selected: VariantCandidate[]; blockers: string[]; };
export type CueScheduleDraft = { cues: CueNode[]; cycles: string[]; overlappingResources: string[]; };
export type CueOwnerCheck = { missingOwners: string[]; pendingAcknowledgments: string[]; conflicts: string[]; };

export type RouteCheck = { eligible: boolean; blockers: string[]; evidenceIds: string[]; };

export type ObservedCueReplay = { actualPath: string[]; unobservedPlans: string[]; conflicts: string[]; latestReceiptTime: string; };

export type LiveShowSources = { production: ImmersiveProduction; scriptEvidence: EvidenceItem[]; productionEvidence: EvidenceItem[]; safetyRules: EvidenceItem[]; sourceRecordStatus: string; priorRecordIds: string[]; reviewAudience: string };

export type CheckpointAuthority = { sourceReady: boolean; priorCheckpointId: string; safetyHold: boolean; holdSignalIds: string[]; authorityReceiptIds: string[]; blockers: string[] };
