import type { State, Node, Edge, Item, Rule, Object, Event } from "../types.js";
export default function reject_step(current: MergeResult, update: Update, step: Step): MergeResult {
return { status: "rejected", text: current.text, applied: current.applied,
         alternatives: [...current.alternatives,
           { update_ids: [update.id], proposal: update.text, reason: step.error }],
         explanation: step.error, base_revision: current.base_revision,
         updates: step.updates, presentation: step.presentation };
}
