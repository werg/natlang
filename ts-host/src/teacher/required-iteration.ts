/** Source-declared root iteration: exactly one completed iterateOn with at least one step. */
export type RequiredRootIteration = { kind: 'one-completed-iterate-on' };

export function checkRequiredRootIteration(spec: RequiredRootIteration | undefined, events: readonly Record<string, unknown>[]): boolean {
  if (spec === undefined) return true;
  if (!spec || spec.kind !== 'one-completed-iterate-on' || Object.keys(spec).length !== 1)
    throw new TypeError('unsupported required root iteration contract');
  const steps = events.filter(event => event.kind === 'iteration_step');
  const ids = new Set(steps.map(event => event.iteration_id));
  if (ids.size !== 1 || [...ids].some(id => typeof id !== 'string' || !id)) return false;
  return steps.some(event => event.phase === 'step') && steps.some(event => event.phase === 'done');
}
