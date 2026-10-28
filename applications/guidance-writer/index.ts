/**
 * The in-run guidance writer (README.md). The natural-language parts are `selectGuidanceFacts.nl` (which visible facts,
 * in what order), `writeGuidance.nl` (the guidance itself, written as Neuralese by the policy model inside the same
 * episode graph) and `planTeacherAttempts.nl`. The crisp parts are the visibility rule and the crisp selection. The
 * natlang runtime loads this folder and runs all of it in-process (ts-host/src/native/training-guidance.ts).
 */
import { visible, visibleFacts } from './visibility.js';
import { CRISP_SELECTION_IDENTITY, crispSelection } from './crispSelection.js';
import type { GuidanceConsumer, GuidanceFact, TeacherAttempts, TeacherGroup } from './types.js';

export type { GuidanceConsumer, GuidanceFact, TeacherAttempts, TeacherGroup };
export { visible, visibleFacts, CRISP_SELECTION_IDENTITY, crispSelection };
