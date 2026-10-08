/** Call records, compilations and their runtime support (plans/TRACE_SPECIALIZATION.md). Platform-neutral part. */
export * from './types.js';
export { snapshot, inputFeatures, tokens, VALUE_CLASSES } from './snapshot.js';
export { normalizeProgram, approachHash, inputLeaves } from './normalize.js';
export { CallCapture, definitionKey, interfaceHash, excluded, type CallStoreLike } from './recorder.js';
export { CompilationCache, caseHashes, caseSources, loadCases, type LoadedCase, type LoadedCompilation } from './compilations.js';
export { Deopt, isDeopt, admit, handoffNote } from './dispatch.js';
