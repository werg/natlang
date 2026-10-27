/** Types of the learned optimizer program (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). Bundles are typed manifests
 * of native references; tensors stay in native artifact storage. */

export type EvidenceDigest = string;

/** One row of the permitted evidence view (exact facts from the runtime). */
export type EvidenceRow = {
  id: string; kind: 'attempt' | 'demonstration' | 'prior-step'; root?: string; passed?: boolean;
  failingBinding?: string | null; errorKind?: string | null; binding?: string;
};

export type EvidenceView = {
  id: string;
  items: EvidenceRow[];
  facts: { roots: number; passed: number; failed: number; failuresByBinding: Record<string, number>; attemptsByBinding: Record<string, number> };
};

/** What `interpretEvidence` returns: where to revise, which evidence shows each problem, and a native digest of the
 * missing behavior per focused binding. */
export type InterpretedEvidence = {
  focus: string[];
  keyEvidence: Record<string, string[]>;
  digests: { binding: string; digest: Neuralese<EvidenceDigest> }[];
};

export type RevisionOperation = 'reencode' | 'replace-skill' | 'weights' | 'keep';
export type RevisionPlan = { operations: Record<string, RevisionOperation>; reason: string };
export type Operator = 'learned-proposal' | 'retrieval' | 'local-refinement' | 'keep';

/** The joint bundle as the program sees it: identities and measured support loss, not tensors. */
export type BundleSummary = { id: string; supportLoss: number; parent: string | null; actor: string };

export type AdaptationState = {
  current: BundleSummary | null;
  incomingLoss: number;
  remainingSteps: number;
  history: { step: number; operator: Operator; bundle: string | null; supportLoss: number | null; accepted: boolean; reason: string }[];
  stopped: boolean;
  stopReason: string;
};

export type AdaptationContext = { program: string; view: EvidenceView; scopes: string[]; maxSteps: number };

export type TrainingMetrics = {
  step: number; labelsStale: boolean; targetsAdmitted: number; proposalsRecorded: number; contrastGroups: number;
  noContrastGroups: number; advantageSnr: number; ladder: Record<string, number | null>; programs: number;
};

export type TrainingWork = 'refit-labels' | 'supervised' | 'policy' | 'collect-proposals' | 'publish-bank' | 'ladder' | 'done';
export type TrainingSchedule = { work: TrainingWork[]; supervisedSteps: number; reason: string };
export type TrainingState = { cycle: number; remainingCycles: number; metrics: TrainingMetrics; log: { cycle: number; work: TrainingWork[]; reason: string }[] };
