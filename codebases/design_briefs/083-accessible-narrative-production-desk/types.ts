export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type AccessibleProduction = RecordMeta & WorkRecord & {
  productionId: string;
  sourceScriptId: string;
  sceneMetadataIds: string[];
  accessibilityOwnerId: string;
  outputFormats: string[];
};
export type NarrativeInformation = RecordMeta & WorkRecord & {
  caseId: string;
  sceneId: string;
  information: string;
  revealTime: number;
  sourceChannel: string;
  importance: string;
};
export type AccessibilityChannelPlan = RecordMeta & WorkRecord & {
  caseId: string;
  informationIds: string[];
  outputChannel: string;
  timingWindow: string;
  competingAudio: string;
};
export type AccessibleSegment = RecordMeta & WorkRecord & {
  caseId: string;
  sceneId: string;
  channel: string;
  text: string;
  startsAt: number;
  endsAt: number;
};
export type AccessibilityCheck = RecordMeta & WorkRecord & {
  caseId: string;
  segmentIds: string[];
  timingIssue: string;
  missingInformation: string;
  reviewerFinding: string;
};
export type AccessibilityPackage = RecordMeta & WorkRecord & {
  caseId: string;
  artifactIds: string[];
  sourceRevision: string;
  reviewRefs: string[];
  deliveryManifestId: string;
};

export type AccessibilityEdit = { instruction: string; sceneEvidence: EvidenceItem[]; styleAndConsent: EvidenceItem[]; requiredProductionIds: string[]; accessibilityEditor: string; priorSegmentIds: string[]; sourceDelta: EvidenceItem[]; playerReceipts?: TargetPlayerReceipt[] };
export type SceneInformation = { sceneId: string; sourceVersion: string; startMs: number | null; endMs: number | null; dialogue: string[]; visibleFacts: string[]; meaningfulSounds: string[]; essentialFactIds: string[]; sourceIds: string[]; unknowns: string[] };
export type RevealBoundary = { factId: string; visibleFromSceneId: string; meaningRevealedFromSceneId: string | null; observableDescription: string; futureInterpretation: string | null; evidenceIds: string[] };
export type ChannelWindow = { id: string; sceneId: string; channel: "caption" | "description" | "transcript"; startMs: number; endMs: number; occupiedBy: string[]; capacityRule: string; sourceVersion: string };
export type AccessibleCue = { id: string; sceneId: string; channel: "caption" | "description" | "transcript"; startMs: number | null; endMs: number | null; text: string; factIds: string[]; sourceVersion: string; speakerEvidenceIds: string[]; unresolved: string[] };
export type SceneEditDependency = { sceneId: string; priorStartMs: number; nextStartMs: number; priorEndMs: number; nextEndMs: number; sourceVersion: string; affectedCueIds: string[] };
export type TargetPlayerReceipt = { receiptId: string; artifactId: string; platform: string; rendererVersion: string; sourceVersion: string; cueIds: string[]; timingPassed: boolean; readingOrderPassed: boolean; formatPassed: boolean; reviewerId: string };
export type AccessibleMediaState = { revision: number; scenes: SceneInformation[]; revealBoundaries: RevealBoundary[]; windows: ChannelWindow[]; cues: AccessibleCue[]; editHistory: SceneEditDependency[]; validatedReceiptIds: string[]; releasedArtifactIds: string[] };
export type AccessibleMediaReview = { mediaState: AccessibleMediaState; proposedCues: AccessibleCue[]; editRequests: string[]; missingEssentialFacts: string[]; staleCueIds: string[]; deliveryArtifactIds: string[]; releaseBlocked: boolean };

export type ChannelAssignment = { sceneId: string; windowId: string | null; channel: "caption" | "description" | "transcript"; factIds: string[]; blockedReason: string | null };
