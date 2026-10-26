import type {
  LegalAction, PersistentWorldState, ReceiptBindingFacts,
  WorldSimulatorReadback, WorldTransitionReceipt,
} from "../../types.js";

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Compare supplied identifiers, revisions, hashes and event bindings; NL owns lifecycle policy. */
export default function validateReceiptBindings(
  world: PersistentWorldState,
  actions: LegalAction[],
  receipts: WorldTransitionReceipt[],
  readback: WorldSimulatorReadback | null,
): ReceiptBindingFacts {
  const blockers: string[] = [];
  const ids = [...new Set(actions.map(action => action.clientActionId))];
  if (ids.length > 1) blockers.push("actions_span_multiple_client_action_ids");
  const canonical = new Map<string, WorldTransitionReceipt>();
  const conflicting = new Set<string>();
  for (const receipt of receipts) {
    const existing = canonical.get(receipt.clientActionId);
    if (!existing) canonical.set(receipt.clientActionId, receipt);
    else if (!same(existing, receipt)) conflicting.add(receipt.clientActionId);
  }
  for (const id of conflicting) blockers.push(`conflicting_duplicate_receipt:${id}`);
  for (const receipt of canonical.values()) {
    if (!ids.includes(receipt.clientActionId)) blockers.push(`receipt_for_unknown_action:${receipt.clientActionId}`);
    if (receipt.worldId !== world.id) blockers.push(`world_id_mismatch:${receipt.clientActionId}`);
    if (receipt.priorRevision !== world.revision) blockers.push(`prior_revision_mismatch:${receipt.clientActionId}`);
  }
  const actionsFacts = ids.map((clientActionId) => {
    const receipt = conflicting.has(clientActionId) ? null : canonical.get(clientActionId) ?? null;
    const owningActions = actions.filter(action => action.clientActionId === clientActionId);
    const facts = {
      clientActionId, receipt, conflictingDuplicate: conflicting.has(clientActionId),
      worldMatches: receipt ? receipt.worldId === world.id : null,
      priorRevisionMatches: receipt ? receipt.priorRevision === world.revision : null,
      actionBound: receipt ? owningActions.length > 0 : null,
      rejectionReasonPresent: receipt?.outcome === "rejected" ? Boolean(receipt.rejectionReason.trim()) : null,
      revisionAdvances: receipt?.outcome === "committed" ? receipt.newRevision > receipt.priorRevision : null,
      readbackWorldMatches: receipt && readback ? readback.worldId === world.id && readback.world.id === world.id : null,
      readbackRevisionMatches: receipt && readback ? readback.revision === receipt.newRevision && readback.world.revision === receipt.newRevision : null,
      readbackRuleVersionMatches: receipt && readback ? readback.ruleVersion === world.ruleVersion && readback.world.ruleVersion === world.ruleVersion : null,
      resourceHashMatches: receipt && readback ? Boolean(receipt.resourceLedgerHash) && receipt.resourceLedgerHash === readback.resourceLedgerHash : null,
      actionReported: receipt && readback ? readback.receiptedActionIds.includes(clientActionId) : null,
      missingReceiptEventIds: receipt && readback ? receipt.eventIds.filter(id => !readback.eventIds.includes(id)) : [],
      invalidEventBindings: [] as string[],
    };
    if (receipt) {
      if (receipt.worldId !== world.id) blockers.push(`world_id_mismatch:${clientActionId}`);
      if (receipt.priorRevision !== world.revision) blockers.push(`prior_revision_mismatch:${clientActionId}`);
      if (owningActions.length === 0) blockers.push(`receipt_for_unknown_action:${clientActionId}`);
      if (receipt.outcome === "rejected" && !facts.rejectionReasonPresent) blockers.push(`rejection_without_reason:${clientActionId}`);
      if (receipt.outcome === "committed" && !facts.revisionAdvances) blockers.push(`nonadvancing_committed_revision:${clientActionId}`);
      if (readback && receipt.outcome === "committed") {
        if (!facts.readbackWorldMatches) blockers.push(`readback_world_mismatch:${clientActionId}`);
        if (!facts.readbackRevisionMatches) blockers.push(`readback_revision_mismatch:${clientActionId}`);
        if (!facts.readbackRuleVersionMatches) blockers.push(`readback_rule_version_mismatch:${clientActionId}`);
        if (!facts.resourceHashMatches) blockers.push(`resource_ledger_hash_mismatch:${clientActionId}`);
        if (!facts.actionReported) blockers.push(`readback_missing_action:${clientActionId}`);
        if (facts.missingReceiptEventIds.length) blockers.push(`readback_missing_receipt_event:${clientActionId}:${facts.missingReceiptEventIds.join(",")}`);
        const events = new Map(readback.world.events.map(event => [event.id, event]));
        facts.invalidEventBindings = receipt.eventIds.filter(eventId => {
          const event = events.get(eventId);
          return !event || event.revision !== receipt.newRevision || !owningActions.some(action => action.id === event.sourceActionId && action.actorId === event.actorId);
        });
        for (const id of facts.invalidEventBindings) blockers.push(`event_action_or_revision_mismatch:${clientActionId}:${id}`);
      }
    }
    return facts;
  });
  return { receipts: [...canonical.values()].filter(receipt => !conflicting.has(receipt.clientActionId)), blockers: [...new Set(blockers)].sort(), actions: actionsFacts };
}
