---
description: Construct nested scene frames before interpreting their events.
args:
  inventory: RecordInventory
returns: FrameIndex
---
1. Call readFrameDeclarations on the original records in inventory.canon and inventory.instructions.
   Consume its declared frames and unresolved ownership findings as the initial frame registry.
   Its contextSourceIds describe declaration sources; they are not an event-processing frontier.
2. Initialize events and localDecisions empty. Build pending from the original SourceRecord objects
   in inventory.selectedScenes and inventory.recollections, deduplicated by their original id.
   Preserve id, path, text and sha256 through this frontier and every named child call.
3. Order pending event interpretation by declared parent availability: outer selected scenes first,
   then memories and quotations within them. For each ready record, call interpretFrame with that
   original record and its inherited FrameDefinition or null for an explicitly outer frame.
4. Consume the child's frames, events, unresolved findings and actual localDecisions. Reconcile a
   repeated frame only when kind, owner and parent agree. Union explicit scene assignments and
   witnesses; a conflict remains unresolved ownership instead of merging incompatible recollections.
5. Each pass removes all interpreted records. If a pass removes none, settle remaining records
   with explicit missing-parent findings. A missing parent never licenses attaching a memory to
   the selected world, and the unchanged pending frontier is not retried indefinitely.
6. Distinguish selected narration, a speaker's memory, a quotation inside that memory and a proposed
   future. Keep those levels even when their descriptions use the same scene or participant names.
7. Deduplicate event identities only within one frame. A selected departure and a remembered
   departure remain separate EventViews unless their own sources explicitly equate their role.
8. Return frames, events, localDecisions and unresolved. Context canon remains available unchanged
   to evaluateAuthority; this branch interprets event records and reconciles discourse ownership.
