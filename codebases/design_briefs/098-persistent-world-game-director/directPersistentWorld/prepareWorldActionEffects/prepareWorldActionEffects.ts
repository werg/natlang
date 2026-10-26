import type { LegalAction, PendingWorldAction, WorldActionEffect } from "../../types.js";

/** Exact serialization for an already-decided effect; lifecycle decisions remain in Natlang. */
export function worldActionEffectKey(
  kind: WorldActionEffect["kind"],
  worldId: string,
  priorRevision: number,
  ruleVersion: string,
  clientActionId: string,
): string {
  return `${kind}:${worldId}:${priorRevision}:${ruleVersion}:${clientActionId}`;
}

/** Exact binding comparison only; Natlang decides whether a mismatch blocks or resumes. */
export function samePendingWorldActionBinding(
  pending: PendingWorldAction,
  worldId: string,
  priorRevision: number,
  ruleVersion: string,
  clientActionId: string,
  actions: LegalAction[],
): boolean {
  return pending.worldId === worldId &&
    pending.priorRevision === priorRevision &&
    pending.ruleVersion === ruleVersion &&
    pending.clientActionId === clientActionId &&
    JSON.stringify(pending.actions) === JSON.stringify(actions);
}
