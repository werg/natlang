/**
 * The kinds of change a run can perform, each as one sentence of instruction. `policy.transformation` names one; the
 * diagnose, hypothesize and edit stages read the sentence as `request.transformation`. The names are the former
 * `reducers/<name>.nl` files; a name is a key here and the host's workflow functions (`repairProgram`, ...) use it.
 */
export const TRANSFORMATIONS:Record<string,string>={
  clarifyInstructions:'Rewrite vague and contradictory instructions into clear executable semantic rules, using concrete input roles and explicit negative or missing-information outcomes. Retain behavior and contracts.',
  extractHelper:'Extract a repeated semantic judgment into a typed named natural-language helper with a companion folder. Retain external behavior and finite control flow.',
  implementProgram:'Implement the requested application in native natlang: named semantic functions and directory reducers, typed contracts, finite exact helpers and folder.iterateOn for bounded improvement. Preserve the pinned external entry.',
  removeDuplicatedGuidance:'Remove duplicated or contradictory instructions. Keep each rule in the narrowest appropriate scope; retain externally required behavior.',
  repairProgram:'Repair a program using supplied failure evidence. Change semantics or structure as justified, preserve regressions and the external entry, and explain the hypothesis tested.',
  simplifyProgram:'Simplify the native semantic program. Remove unnecessary machinery, make semantic decisions explicit in natlang, retain finite execution and external behavior.',
  specializeForModel:'Simplify instructions and API usage for an underpowered coding model using observed failures. Prefer explicit folder.propose, evaluator.check/evaluate, folder.accept and folder.select. Base each change on observed failures so it holds across cases.',
};

/** The instruction sentence of a named transformation; an unknown name lists the known ones. */
export function transformationText(name:string):string {
  const text=TRANSFORMATIONS[name];
  if(text===undefined)throw Error('A transformation is one of: '+Object.keys(TRANSFORMATIONS).join(', ')+'.');
  return text;
}
