/**
 * The in-run guidance writer (README.md). The natural-language parts are `selectGuidanceFacts.nl` (which visible facts,
 * in what order), `writeGuidance.nl` (the guidance itself, written as Neuralese by the policy model inside the same
 * episode graph), `writeHint.nl` and `planTeacherAttempts.nl`. The crisp parts are the visibility rule, the crisp selection, the
 * trajectory overview and the hint check. The
 * natlang runtime loads this folder and runs all of it in-process (ts-host/src/native/training-guidance.ts).
 */
import { visible, visibleFacts } from './visibility.js';
import { CRISP_SELECTION_IDENTITY, crispSelection } from './crispSelection.js';
import { trajectoryOverview } from './trajectoryOverview.js';
import { checkGuidance, checkHint, HINT_BUDGET_TOKENS } from './hintCheck.js';
import type { GuidanceConsumer, GuidanceFact, TeacherAttempts, TeacherGroup, TrajectoryCall, TrajectoryOverview,
  TrajectoryStep } from './types.js';

export type { GuidanceConsumer, GuidanceFact, TeacherAttempts, TeacherGroup, TrajectoryCall, TrajectoryOverview, TrajectoryStep };
export { visible, visibleFacts, CRISP_SELECTION_IDENTITY, crispSelection, trajectoryOverview, checkGuidance, checkHint, HINT_BUDGET_TOKENS };
