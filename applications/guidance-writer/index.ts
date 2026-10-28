/**
 * The in-run guidance writer (README.md). The natural-language parts are `selectGuidanceFacts.nl` (which visible facts,
 * in what order) and `writeGuidance.nl` (the guidance itself, written as Neuralese by the policy model inside the same
 * episode graph). The crisp parts are the visibility rule and the crisp selection. The natlang runtime loads this folder
 * and runs all of it in-process (ts-host/src/native/training-guidance.ts).
 */
export type * from './types.js';
export { visible, visibleFacts } from './visibility.js';
export { CRISP_SELECTION_IDENTITY, crispSelection } from './crispSelection.js';
