/**
 * The evaluation and adaptation APIs a host script uses to run natlang suites in Node: `@natlang/node/evaluation`,
 * `/adaptation` and `/optimize` as one entry point (staged for Vibestudio by scripts/stage-vibestudio.mjs).
 */
export * from '../evaluation/index.js';
export * from '../adaptation/index.js';
export { loadAdaptation } from '../adaptation/node.js';
export { optimize, resumeOptimization, exportAdaptationPatch, revalidateAdaptation } from '../optimization/index.js';
export { executorIdentityForChoice } from '../model/config.js';
// The model session a suite's executor runs on: the profile it names, opened as the CLI opens it (stopped by `close`).
export { loadModelConfiguration, createResolvedModelSession, localModelPrerequisites, wrapDriver,
  // A session as the driver `evaluate`/`optimize` take: its turns, decision scorer, context window and native runtime.
  sessionDriver } from '../model/index.js';
export { createNatlangRuntime, nodePlatform } from '../runtime/node.js';
