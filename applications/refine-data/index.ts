/**
 * Data generation for refinement types (plans/REFINEMENT_DATA.md): the teacher-side stages of the `refine-judge`
 * family. `exemplify` writes satisfying values for a predicate, `nearMiss` edits one into a minimal violation and
 * `verifyNearMiss` is the independent second pass. The stages run through the normal runtime (so their calls are
 * recorded) from ts-host/scripts/refine-data/run-stage.mjs; scripts/refine_data.py turns their output into rows.
 */
import exemplify from './exemplify.nl';
import nearMiss from './nearMiss.nl';
import verifyNearMiss from './verifyNearMiss.nl';

export type * from './types.js';
export { exemplify, nearMiss, verifyNearMiss };
