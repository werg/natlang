---
description: Own one ready window and commit only the semantic deltas returned by its children.
args:
  state: WindowState
returns: WindowState
---
1. Read actual pending slots, submitted response order and the compiled timing protocol. Select the next
   permitted active slot: outstanding submitted interrupts precede their suspended announcement;
   eligible after-batch consequences follow their source priority. Preserve a missing decisive ordering
   choice as unresolved and finished. Independent source-commuting order may use a stable ID order.
2. Copy the current board as beforeBoard. If the selected slot is a response, call resolveInterrupts
   with its actual IntentReading, beforeBoard and source. Consume only that returned response delta;
   preserve captured announcement fields unchanged. Record a response WindowTrace with no event batch.
3. For an announcement or queued trigger, call prepareBundle with this slot, the current state and
   beforeBoard. The result owns actual effect parts and bound recipients. Pass it to rewriteBundle
   with beforeBoard, source and protocol; consume its returned bundle and replacement history.
4. Call resolveBatch on that actual rewritten result and the same immutable beforeBoard. Its proposed
   object/resource/departure changes must resolve together; an effect part cannot observe a sibling's
   later death or return. Keep the returned BatchDelta as the committed semantic decision.
5. If any selected child reports a decisive unresolved field, retain it and finish with the original
   beforeBoard for this uncommitted window. No partially selected response or batch delta is applied.
6. Apply the chosen complete delta mechanically: check each declared before value against beforeBoard,
   replace/add the specified object and resource rows, and mark the selected source token IDs spent.
   Preserve all unrelated fields. Assign a fresh snapshot revision identifying this committed window.
7. After a resolved event batch, call collectTriggers with the actual bundle, replacements, BatchDelta,
   beforeBoard and afterBoard, plus this actual state. Consume every returned decision,
   including ineligible ones. Spend only newly emitted trigger tokens and activate their existing slots.
8. Append a WindowTrace copying sourceIdentity from the selected WorkSlot and containing actual
   before/after boards, bundle, replacement decisions, batch
   and trigger judgments. For a trigger window, retain its causal parent slot from the originating
   batch; its consequence never becomes an unsubmitted player announcement.
9. Mark this selected slot consumed and remove its identity once from remainingWorkSlotIds and pending
   work. Add newly activated identities only when already inventoried and not consumed. Preserve prior
   traces and all typed queued TriggerDecisions; they fix recipients even if later objects change.
10. Set finished when no active permitted work remains. Otherwise return the next state with a strictly
    smaller remaining-work measure. Event meaning, ordering and applicability came from NL judgments;
    applying their already chosen field deltas is the exact commit seam.
