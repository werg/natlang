/** Adapted from Ax b780a14a3cb94d5ac572db04038399aef655c76c. Copyright Ax contributors. Apache-2.0. Modified for native keys and deterministic randomness; see vendor/ax-gepa/CHANGES.md. */
import type { ComponentTarget } from './gepaSelection.js';

export function getUpdateGroup(
  target: Readonly<ComponentTarget>,
  targets: readonly ComponentTarget[]
): ComponentTarget[] {
  const byId = new Map(targets.map((item) => [item.key, item]));
  const out = new Map<string, ComponentTarget>();
  const visit = (id: string): void => {
    const item = byId.get(id);
    if (!item || out.has(id)) return;
    out.set(id, item);
    for (const dep of item.dependsOn ?? []) visit(dep);
  };
  visit(target.key);
  return [...out.values()];
}
