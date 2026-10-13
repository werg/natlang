---
description: Construct nested scene frames before interpreting their events.
args:
  inventory: RecordInventory
returns: FrameIndex
---
1. Read selectedScenes and recollections, plus explicit frame definitions in canon and instructions.
   Inventory all frame IDs and declared parent relations before interpreting event content.
2. Initialize frames, events, localDecisions and unresolved empty. Order interpretation by parent availability:
   outer selected frames first, then their memories or quotations, then deeper quoted recollections.
3. Process the finite pending source records. For a record whose parent is already known, call
   interpretFrame(record, parentDefinitionOrNull). Consume the returned frames and events and append their actual localDecisions.
4. Reuse a declared frame identity only if kind, owner and parent agree. A conflict is unresolved
   ownership and remains visible; it cannot merge two incompatible recollections into one event.
5. Each pass removes all records interpreted with known parents. If a pass removes none, settle
   remaining records with explicit missing-parent findings instead of retrying the same frontier.
6. Distinguish selected narration, a speaker's memory, a quotation inside that memory and a proposed
   future. Keep those levels even when their descriptions use the same scene or participant names.
7. Deduplicate identical event identities only within the same frame. A selected departure and a
   remembered departure are separate EventViews unless their source explicitly equates their role.
8. Return the FrameIndex. Its unresolved entries name the source records and missing frame premises.
   The child interprets content; this function owns only frame ordering and identity reconciliation.
