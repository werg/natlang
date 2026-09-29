import type { AxGEPAComponentTarget } from './gepaComponents.js';

export function getGEPAUpdateGroup(
  target: Readonly<AxGEPAComponentTarget>,
  targets: readonly AxGEPAComponentTarget[]
): AxGEPAComponentTarget[] {
  const byId = new Map(targets.map((item) => [item.id, item]));
  const out = new Map<string, AxGEPAComponentTarget>();
  const visit = (id: string): void => {
    const item = byId.get(id);
    if (!item || out.has(id)) return;
    out.set(id, item);
    for (const dep of item.dependsOn ?? []) visit(dep);
  };
  visit(target.id);
  return [...out.values()];
}
