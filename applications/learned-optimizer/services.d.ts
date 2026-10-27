import type { BundleSummary, EvidenceDigest, EvidenceView, RevisionOperation, TrainingMetrics } from './types.js';

/** Exact adaptation services (training/neuralese/natlang_neuralese/tune/learned/apply.py). They score, write and
 * compose; they never decide what to revise. */
export type OptimizerService = {
  /** The permitted support view and the incoming bundle's exact support loss. */
  evidence(): Promise<{ view: EvidenceView; scopes: string[]; incomingLoss: number }>;
  /** The learned writer's complete bundle from `parent` (null: the incoming bundle), revising exactly what
   * `operations` selects. `sample: false` decodes deterministically. Focus, key evidence and digests enter the
   * writer's request natively. */
  propose(request: { parent: string | null; sample: boolean; seed: number; operations: Record<string, RevisionOperation>;
    focus: string[]; keyEvidence: Record<string, string[]>; digests: { binding: string; digest: Neuralese<EvidenceDigest> }[] }): Promise<BundleSummary>;
  /** Local in-bank refinement of the selection from `parent` by gradient on the support evidence. */
  refine(request: { parent: string | null; steps: number; operations: Record<string, RevisionOperation> }): Promise<BundleSummary>;
  /** Gradient-free mixing over the selection's bank views from `parent` (the retrieval baseline). */
  retrieve(request: { parent: string | null; operations: Record<string, RevisionOperation> }): Promise<BundleSummary>;
};

/** Exact outer-training services (trainer.py, fit.py, ladder.py). */
export type TrainerService = {
  metrics(): Promise<TrainingMetrics>;
  refitLabels(): Promise<{ targets: number }>;
  supervised(request: { steps: number }): Promise<{ step: number }>;
  collectProposals(request: { perContext: number }): Promise<{ recorded: number }>;
  policy(): Promise<{ step: number; estimator: string }>;
  publishBank(): Promise<{ snapshot: string }>;
  ladder(): Promise<Record<string, number | null>>;
};

declare module 'natlang:services' {
  export const optimizer: OptimizerService;
  export const trainer: TrainerService;
}
