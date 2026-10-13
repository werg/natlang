---
description: Materialize one announced or queued consequence using its correct captured and live references.
args:
  slot: WorkSlot
  state: WindowState
  beforeBoard: BoardSnapshot
returns: EventBundle
---
1. Resolve this active slot against actual captured announcements or actual queued TriggerDecisions.
   For an announcement, use its original part IDs, printed order, quantities and tracked instance IDs.
   For a trigger, use the consequenceParts and recipient already selected by its qualifying decision.
2. Join each tracked instance with beforeBoard. Preserve card identity separately from instance identity.
   A departed instance's missing live target is a source-defined resolution question, not permission to
   substitute another stay of the same card. Carry that question forward with its exact evidence.
3. Interpret the effect kind from the actual source: damage, growth, ink, return, control or attachment.
   Materialize only the fields required for that kind. A return part carries the source-defined new
   BoardObject; a resource part carries its already bound recipient; an attachment names its target.
4. Preserve each original lineagePartId through replacements. Initialize appliedRuleIds empty only for
   a genuinely new part lineage. The bundle shares one beforeRevision and carries the announcement's
   actual ReferenceBindings so later judgments can distinguish captured from current control.
5. Return bundleId, causeSlotId, intentId, beforeRevision, parts, references and exact witnesses. The
   caller passes this object unchanged into replacement interpretation, which owns any later rewrite.
