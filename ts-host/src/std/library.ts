/**
 * `@natlang/std` (`@natlang/node/std`, `@natlang/browser/std`): semantic combinators. TypeScript mechanism around named
 * judgments; the judgments the library supplies are `.nl` functions shipped with it (ts-host/std/*.nl) and exported
 * here so a program can call, decide, rebind, escalate, trace and specialise them like its own.
 * Design: ts-host/std/DECOMPOSITION.md. Its exports, as programs import them (index.ts registers them for virtual programs).
 */
export { semanticMap, semanticFilter, semanticPartition, semanticReduce, semanticMatch, semanticRank } from './collections.js';
export type { CarryMode, MapOptions, FilterOptions, PartitionOptions, Partition, ReduceOptions, MatchOptions, MatchPair, MatchResult,
  RankOptions, RankedEntry, Ranking } from './collections.js';
export { semanticRepair, fixedPoint, semanticMerge } from './converge.js';
export type { Problems, RepairOptions, Repair, RepairAttempt, Revision, FixedPointOptions, FixedPoint, MergeOptions, Merge,
  MergeConflict, Merged, Unresolved } from './converge.js';
export { escalate, withContext, compileRule } from './policy.js';
export type { EscalateOptions, Escalated, RuleExample, CompileRuleOptions } from './policy.js';
export { proposeAndVerify, branchAndSelect } from './workspace.js';
export type { Workspace, Reducer, ProposeOptions, Proposed, Branch, Candidate, SelectOptions, Selected } from './workspace.js';
export { intentOf, compatible, combine, settled, beats, meets, relates, compileRuleFn, shouldEscalate, stepKind } from './judgments.js';
export { instructionsOf, stepsOf } from './instructions.js';
export type { StepKind, KindedStep, Instructions } from './instructions.js';
export type { Intent, Edit, EscalationHistory, TierCost } from './judgments.js';
