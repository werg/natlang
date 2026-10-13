---
description: Set a finite source-governed replacement frontier and consume its rewritten parts.
args:
  bundle: EventBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
  protocol: WindowProtocol
returns: RewrittenBundle
---
1. Inventory standing and replacement rules relevant to these actual effect kinds and original part
   lineages. Include potentially inapplicable rules so their negative judgments can explain the result;
   context definitions and after-resolution triggers belong to separate semantic responsibilities.
2. Call resolveOrdering(bundle, beforeBoard, source, protocol) to obtain source-governed applications.
   Each application pairs one rule with one original lineage and has one finite work identity. A
   source-defined once-per-lineage rule cannot acquire another use merely by rewriting that lineage.
3. If ordering is decisively unresolved, return the original bundle with that finding and no committed
   token consumption. Otherwise initialize ReplacementState with the actual ordered applications,
   beforeBoard, empty decisions/reservations and this bundle, retaining source and protocol.
4. Invoke applyReplacement.iterateOn with measure remainingApplications.length until the list is empty
   or unresolved prevents a supported rewrite. Each step consumes one application and returns the
   actual updated bundle plus its semantic decision. Shared token reservations carry between steps.
5. Consume the completed bundle, ordered decisions and reserved token IDs. Validate that each output
   part preserves its declared original lineage and that every quantity is a source-supported nonnegative
   integer. These are output integrity checks after semantic interpretation, not replacement selection.
6. Return RewrittenBundle. Every before/after decision records the current rewritten input, so later
   trigger interpretation can distinguish announced damage, post-prevention damage and converted growth.
