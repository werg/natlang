/**
 * Scripted answers for the authored program improver's stages. One experiment runs the step (the improveStep.nl call that
 * runs lifecycle.step), then diagnose, hypothesize and the edit; each opens with its own text.
 */
export const STAGES = {
  diagnose: 'Say what the training evidence shows about the program',
  hypothesize: 'Choose the one edit to the program source that the diagnosis supports',
  edit: 'Make the edit that request.hypothesis describes',
};
export const STEP = 'return await lifecycle.step(folder,evaluator,plans,state,policy);';
export const DIAGNOSE = 'return {observations:[],pattern:"scripted diagnosis"};';
export const HYPOTHESIZE = 'return {kind:"instruction",statement:"scripted hypothesis",files:[],predictedChange:"quality rises"};';
const text = (value, opening) => typeof value === 'function' ? value(opening) : value;

/**
 * A scripted model for the stages: `edit` is the edit stage's eval code (it returns {summary, preserves}); the other
 * stages default to a fixed diagnosis, a fixed hypothesis and the step. `other(opening)` answers any other opening first.
 */
export function stagedImprover({ edit, diagnose = DIAGNOSE, hypothesize = HYPOTHESIZE, step = STEP, other } = {}) {
  return opening => {
    const custom = other?.(opening);
    if (custom !== undefined && custom !== null) return custom;
    if (opening.includes(STAGES.diagnose)) return text(diagnose, opening);
    if (opening.includes(STAGES.hypothesize)) return text(hypothesize, opening);
    if (opening.includes(STAGES.edit)) return text(edit, opening);
    return text(step, opening);
  };
}
/** The model driver that completes every eval with its value, as the improver's directory reducers do. */
export const finishing = model => async (request, signal) => {
  const turn = await model.driver(request, signal);
  for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
  return turn;
};
