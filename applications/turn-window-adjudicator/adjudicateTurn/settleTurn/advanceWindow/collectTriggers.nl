---
description: Judge consequences from the actual resolved batch and activate only their source-inventoried slots.
args:
  bundle: EventBundle
  replacements: ReplacementDecision[]
  batch: BatchDelta
  beforeBoard: BoardSnapshot
  afterBoard: BoardSnapshot
  state: WindowState
returns: TriggerCollection
---
1. Select the source trigger rules whose stated event scope can be tested against this actual bundle.
   On the announced batch, account for every printed trigger, including exhausted or unmet abilities.
   Keep earlier queued decisions unchanged; eligibility at another window does not rebind their recipients.
2. For each selected trigger call interpretTrigger with its RuleReading, actual bundle/replacements,
   batch, beforeBoard, afterBoard and the captured announcements. Consume that actual returned decision.
3. For eligible true, locate its already inventoried one-use slot and verify it is neither consumed nor
   already active. Its token IDs must still be available in the committed afterBoard. Reserve them once
   across this collection; a repeated rule cannot allocate a fresh capacity by changing a trace ID.
4. Add the typed decision's consequence parts and recipient to the queue unchanged. Activate its slot,
   spend its returned token IDs when queued, and carry its source priority. Ordering later reads these
   actual decisions rather than recomputing eligibility after a preceding consequence resolves.
5. Preserve eligible false with its specific failed condition and zero consumed tokens. Preserve null
   with the exact missing premise as unresolved; a consequence requiring that premise stays uncommitted.
6. Return all decisions, activatedSlotIds, consumedTokenIds and unresolved. Negative quantity thresholds,
   converted non-damage events and exhausted capacity are distinct explanations, each grounded in source.
