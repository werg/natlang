/** Exact root-side effects declared by a task's host-only oracle. */
export type RequiredAction = { kind: 'read_code'; name: string };

export function checkRequiredActions(spec: RequiredAction[] | undefined,
  rootCallId: string, trace: Record<string, unknown>[]): boolean {
  if (spec === undefined) return true;
  if (!Array.isArray(spec) || !spec.length) throw new TypeError('required_actions must be a nonempty list');
  return spec.every(item => {
    if (!item || item.kind !== 'read_code' || typeof item.name !== 'string' || !item.name ||
        Object.keys(item).some(key => !['kind', 'name'].includes(key)))
      throw new TypeError('required action needs an exact read_code name');
    return trace.some(event => event.kind === 'action' && event.call_id === rootCallId &&
      event.name === 'read_code' && event.outcome === 'ok' &&
      !!event.arguments && typeof event.arguments === 'object' &&
      (event.arguments as Record<string, unknown>).name === item.name);
  });
}
