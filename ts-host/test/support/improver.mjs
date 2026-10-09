/**
 * Scripted answers for the authored program improver's stages. One experiment is crisp (improveStep/lifecycle.ts) and calls
 * the diagnose, hypothesize and edit stages; each opens with its own text.
 */
export const STAGES = {
  diagnose: 'Say what the training evidence shows about the program',
  hypothesize: 'Choose the one edit to the program source that the diagnosis supports',
  edit: 'Make the edit that request.hypothesis describes',
};
export const DIAGNOSE = 'return {observations:[],pattern:"scripted diagnosis"};';
export const HYPOTHESIZE = 'return {kind:"instruction",statement:"scripted hypothesis",files:[],predictedChange:"quality rises"};';
const text = (value, opening) => typeof value === 'function' ? value(opening) : value;

/**
 * A scripted model for the stages: `edit` is the edit stage's eval code (it returns {summary, preserves}); the other
 * stages default to a fixed diagnosis and a fixed hypothesis. `other(opening)` answers any other opening first.
 */
export function stagedImprover({ edit, diagnose = DIAGNOSE, hypothesize = HYPOTHESIZE, other } = {}) {
  return opening => {
    const custom = other?.(opening);
    if (custom !== undefined && custom !== null) return custom;
    if (opening.includes(STAGES.diagnose)) return text(diagnose, opening);
    if (opening.includes(STAGES.hypothesize)) return text(hypothesize, opening);
    if (opening.includes(STAGES.edit)) return text(edit, opening);
    return undefined;
  };
}
/** The model driver that completes every eval with its value, as the improver's directory reducers do. */
export const finishing = model => async (request, signal) => {
  const turn = await model.driver(request, signal);
  for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
  return turn;
};

/**
 * The authored improver with its experiment step replaced: `body` is the body of
 * `step(folder, evaluator, plans, state, policy)` (improveStep/lifecycle.ts), a crisp directory reducer that may call the
 * stages `diagnose`, `hypothesize`, `editSource` and `editSourceStructural`. Returns the frozen source for `improverSource`.
 */
export async function improverWithStep(body) {
  const { Folder } = await import('../../dist/index.js');
  const { AUTHORED_IMPROVER } = await import('../../dist/improvement/authored-source.js');
  const authored = Folder.fromFiles(AUTHORED_IMPROVER);
  authored.writeText('improveStep/lifecycle.ts', [
    "import diagnose from './diagnose.nl';", "import hypothesize from './hypothesize.nl';",
    "import editSource from './editSource.nl';", "import editSourceStructural from './editSourceStructural.nl';",
    'export async function step(folder: any, evaluator: any, plans: any, state: any, policy: any): Promise<any> {', body, '}', ''].join('\n'));
  return authored.snapshot();
}
