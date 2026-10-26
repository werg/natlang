
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type ProblemStudy = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; question: string; populationDefinition: string; windowStart: string; windowEnd: string; ownerId: string };
export type ProblemEpisode = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; customerId: string; situation: string; desiredOutcome: string; observedObstacle: string; occurredAt: string };
export type IncidentGroup = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; episodeIds: string[]; customerId: string; incidentKey: string; independenceBasis: string };
export type ProblemConcept = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; name: string; definition: string; episodeIds: string[]; boundaryExamples: string[] };
export type TrendSlice = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; conceptId: string; cohortQueryId: string; numerator: number; denominator: number; qualification: string };
export type InvestigationBrief = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; conceptIds: string[]; question: string; discriminatingEvidence: string; ownerId: string };
export type DraftProblemEpisode = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; customerId?: string; situation?: string; desiredOutcome?: string; observedObstacle?: string; occurredAt?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type ObservatoryQuestion = { caseId: string; studyQuestion: string; targetPopulation: string; intervalStart: string; intervalEnd: string; requestedBy: string; expectedSnapshotVersion: string; stableConceptIds: string[] };
export type SignalObservationSet = { snapshotVersion: string; studies: ProblemStudy[]; conversationAndSurveyPages: EvidencePage[]; exposureFrames: EvidencePage[]; identityMappingEvents: EvidencePage[]; consentEvents: EvidencePage[]; channelCoverage: EvidencePage[]; policies: PolicyCandidate[]; searchedPartitions: string[] };

export type CustomerEpisodeContext = { customerId: string; attemptedJob: string; desiredOutcome: string; obstacle: string; consequence: string; featureRequest: string; occurredAt: string; sourceIds: string[]; attributionAllowed: boolean; uncertainty: string[] };

export type CustomerObservatoryRequest = { caseId: string; question: string; populationDefinition: string; windowStart: string; windowEnd: string; ownerId: string; expectedVersion: string };
export type CustomerSignalDossier = { snapshotVersion: string; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; studies: ProblemStudy[]; evidencePages: EvidencePage[]; rawEpisodes: CustomerEpisodeContext[]; outageReceipts: EvidencePage[]; samplingFrames: EvidencePage[]; trendReceipts: TrendAggregateReceipt[]; consentChanges: EvidencePage[]; identityMapVersions: EvidencePage[]; conceptRenameEvents: EvidencePage[] };
export type TrendAggregateReceipt = { conceptId: string; window: string; cohortQueryId: string; numerator: string; denominator: string; collectionChannelVersion: string; exposureVersion: string; queryHash: string; aggregationEngine: string; completeness: string; evidenceRefs: string[] };
export type EpisodeRegister = { situations: CustomerEpisodeContext[]; featureRequestsSeparated: string[]; unsupportedSeverity: string[]; rawSourceIds: string[]; missingContext: string[] };
export type IncidentDedupLedger = { incidentGroups: IncidentGroup[]; mergedMessageIds: string[]; independentRepeatIds: string[]; uncertainIdentityIds: string[]; eventVolumeOnlyIds: string[] };
export type StableProblemMap = { concepts: ProblemConcept[]; stableConceptIds: string[]; conceptVersions: string[]; reclassifiedEpisodeIds: string[]; counterexamples: string[]; renameHistory: string[] };
export type ComparableTrendSet = { slices: TrendSlice[]; excludedWindows: string[]; brokenComparability: string[]; exactReceiptRefs: string[]; unsupportedRateClaims: string[] };
export type InvestigationQueue = { briefs: InvestigationBrief[]; unresolvedMechanisms: string[]; supportingAndOpposingRefs: string[]; excludedFeatureRequests: string[] };
export type ConsentDerivativePlan = { withdrawnCustomerIds: string[]; attributableQuoteIdsToRemove: string[]; derivedArtifactsToRecompute: string[]; evidenceRefs: string[] };
export type IdentityVersionLedger = { acceptedMappingVersions: string[]; ambiguousCustomerIds: string[]; countsBlockedUntilVersioned: string[]; evidenceRefs: string[] };

export type ObservatoryFrame = { question: string; population: string; windowStart: string; windowEnd: string; ownerId: string; studyVersion: string; blockers: string[] };
export type ObservatoryCheckpoint = { caseId: string; revision: number; snapshotVersion: string; sourceVersions: string[]; episodes: EpisodeRegister[]; identityHistory: IdentityVersionLedger[]; incidentEditions: IncidentDedupLedger[]; conceptEditions: StableProblemMap[]; trendEditions: ComparableTrendSet[]; investigationHistory: InvestigationQueue[]; consentActions: string[]; unresolvedEffects: string[] };
export type ObservatoryInvalidation = { revokedDerivativeIds: string[]; staleTrendKeys: string[]; preservedTrendKeys: string[]; reopenedInvestigationKeys: string[]; unresolvedDeletionReceiptIds: string[]; reasons: string[] };
export type ObservatoryRun = { caseId: string; checkpoint: ObservatoryCheckpoint; frame: ObservatoryFrame; episodes: EpisodeRegister; identityLedger: IdentityVersionLedger; consentPlan: ConsentDerivativePlan; incidents: IncidentDedupLedger; problemMap: StableProblemMap; trends: ComparableTrendSet | null; investigations: InvestigationQueue; invalidations: ObservatoryInvalidation; outstandingEffects: string[]; commitAllowed: boolean };
export type ConsentEffectReview = { pendingRemovalRefs: string[]; pendingRecomputeRefs: string[]; preservedNonAttributableIds: string[]; sourceEvidenceRefs: string[] };
export type TrendLineageReview = { staleTrendKeys: string[]; preservedTrendKeys: string[]; editionsToRetain: ComparableTrendSet[]; blockedWindows: string[] };
