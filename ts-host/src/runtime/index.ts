/** Platform-neutral natlang runtime API. Platform entry points install a context store and evaluator. */
import * as lowered from './lowered.js';
export { createNatlangRuntime, NatlangRuntime, NatlangTask, recordingServices, resolveFrame,
  setDefaultEnvironmentFactory } from './runtime.js';
export type { ModelDriver, ModelConfig, NatlangRuntimeOptions, NatlangLimits, TaskOptions, TraceSink,
  InvocationTrace, Services } from './runtime.js';
export { modelTurnsSoFar } from '../native/agent.js';
export { NatlangContextError, NatlangRecursionError, SlotContextStore, setContextStore, currentFrame } from './context.js';
export type { ContextStore, Frame } from './context.js';
export { NatlangCallError, invokeDefinition } from './kernel.js';
export type { CallableDefinition, CaptureCell } from './kernel.js';
export { isNatlangCallable, namedCallable, callableTree, defineNatlang } from './callable.js';
export { Iteration, IterationDivergedError, IterationStepError, IterationLimitError, MemoryIterationStatistics,
  defaultProgressJudge } from './iterate.js';
export type { IterationStatisticsStore, SiteStatistics, ProgressJudgeFunction, StepRecord } from './iterate.js';
export { nl, iterateOn } from './surface.js';
export type { NatlangFunction, NlResult, IterationEvent, IterationTrajectory, ProgressVerdict } from './surface.js';
/** Support functions targeted by compiled modules. Not an application API. */
export const __natlang = lowered;
