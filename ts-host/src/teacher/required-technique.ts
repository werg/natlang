/**
 * The technique a hinted twin requires. A `:hinted` curriculum twin (inline-curriculum `hinted()`) appends a hint to its
 * request: "Required: solve this with iterateOn ..." when the case's `iterate` is `required`, and "Required: make each
 * judgment about an item with a natural-language function ..." when its `inline` is `required`. The answer oracle grades
 * the result only, so a run that typed the answer by hand scored the same as one that followed the hint, and the twin
 * taught nothing. The requirement is enforced on the actual run: an iterateOn that took steps and finished, and at least
 * one inline natural-language call. Unhinted twins keep both as notes (they were never told).
 */
export type RequiredTechnique = { iterate: boolean; inline: boolean };
export type TechniqueResult = { iterate_on?: boolean; inline_nl?: boolean };

export function hintedTechnique(curriculum: unknown): RequiredTechnique | undefined {
  const c = curriculum as { hinted_of?: unknown; hint?: unknown; iterate?: unknown; inline?: unknown } | undefined;
  if (!c || typeof c.hinted_of !== 'string' || typeof c.hint !== 'string') return undefined;
  const iterate = c.iterate === 'required' && c.hint.includes('Required: solve this with iterateOn');
  const inline = c.inline === 'required' && c.hint.includes('Required: make each judgment about an item with a natural-language function');
  return iterate || inline ? { iterate, inline } : undefined;
}

/** `rootEvents` are the root call's trace; `childEvents` one event list per nl child call the run made. */
export function checkHintedTechnique(required: RequiredTechnique, rootEvents: readonly Record<string, unknown>[],
    childEvents: readonly (readonly Record<string, unknown>[])[]): TechniqueResult {
  const result: TechniqueResult = {};
  if (required.iterate) {
    const steps = rootEvents.filter(event => event.kind === 'iteration_step');
    result.iterate_on = steps.some(event => event.phase === 'step') && steps.some(event => event.phase === 'done');
  }
  if (required.inline)
    result.inline_nl = childEvents.some(events => events.some(event => event.kind === 'manifest' && event.inline_instruction_site));
  return result;
}

export const techniqueAccepted = (result: TechniqueResult): boolean => Object.values(result).every(value => value === true);
