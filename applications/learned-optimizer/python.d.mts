import type { BundleSummary, EvidenceView, RevisionOperation } from './types.js';

/** How to run the Python entry: an argv ending in `-m` (e.g. `[".venv/bin/python", "-m"]`) and its environment. */
export type PythonEntry = { readonly argv: readonly string[]; readonly env?: Readonly<Record<string, string>>; readonly cwd?: string };

export declare class PythonService {
  constructor(entry: PythonEntry);
  request(body: Record<string, unknown>): Promise<unknown>;
  command(command: string, spec: Record<string, unknown>): Promise<unknown>;
  close(): Promise<void>;
}

export type Bundle = { id: string; parent: string | null; provenance?: { actor?: string } };

export declare class AdaptationServices {
  constructor(python: PythonService, plan: string, out: string);
  bundle(id: string): Bundle;
  readonly optimizer: {
    evidence(): Promise<{ view: EvidenceView; scopes: string[]; incomingLoss: number }>;
    propose(request: { parent: string | null; sample: boolean; seed: number; operations?: Record<string, RevisionOperation>;
      focus: string[]; keyEvidence?: Record<string, string[]>; digests: { binding: string; digest: unknown }[] }): Promise<BundleSummary>;
    refine(request: { parent: string | null; steps: number; operations?: Record<string, RevisionOperation> }): Promise<BundleSummary>;
    retrieve(request: { parent: string | null; operations?: Record<string, RevisionOperation> }): Promise<BundleSummary>;
  };
  supportLoss(id: string | null): Promise<number>;
  release(id: string | null): Promise<Record<string, unknown>>;
}

export declare const optimizerDeclaration: string;
