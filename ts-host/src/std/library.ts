/**
 * `@natlang/std` (`@natlang/node/std`, `@natlang/browser/std`): semantic combinators. TypeScript mechanism around named
 * judgments; the judgments the library supplies are `.nl` functions shipped with it (ts-host/std/*.nl) and exported
 * here so a program can call, decide, rebind, escalate, trace and specialise them like its own.
 * Design: ts-host/std/DECOMPOSITION.md. Its exports, as programs import them (index.ts registers them for virtual programs).
 */
export { semanticRepair, fixedPoint, semanticMerge } from './converge.js';
export type { Problems, RepairOptions, Repair, RepairAttempt, Revision, FixedPointOptions, FixedPoint, MergeOptions, Merge,
  MergeConflict, Merged, Unresolved } from './converge.js';
export { proposeAndVerify, branchAndSelect } from './workspace.js';
export type { Workspace, Reducer, ProposeOptions, Proposed, Branch, Candidate, SelectOptions, Selected } from './workspace.js';
export { intentOf, compatible, combine, settled, beats } from './judgments.js';
export type { Intent, Edit } from './judgments.js';
