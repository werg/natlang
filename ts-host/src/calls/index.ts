/** Call records, compilations and their runtime support (plans/TRACE_SPECIALIZATION.md). Platform-neutral part. */
export * from './types.js';
export { snapshot, inputFeatures, tokens, VALUE_CLASSES } from './snapshot.js';
export { normalizeProgram, approachHash, inputLeaves } from './normalize.js';
export { CallCapture, definitionKey, interfaceHash, excluded, type CallStoreLike } from './recorder.js';
export { CompilationCache, caseHashes, caseServices, caseSources, loadCases, type LoadedCase, type LoadedCompilation } from './compilations.js';
export { Deopt, isDeopt, admit, handoffNote } from './dispatch.js';
export { hostCaptures, type HostCapture, type CaptureSource } from './host-capture.js';
export { TierEngine, TierLedger, ModelTier, CompiledTier, CrispTier, NeuraleseTier, ladderOf, runTiered, type Tier, type TierInput, type TierSettings,
  type TierAttempt, type TierEvent, type CrispImplementation } from './tiers.js';
