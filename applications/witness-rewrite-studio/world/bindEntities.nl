---
description: Bind explicit aliases while preserving distinct people and objects.
args:
  inventory: RecordInventory
  frames: FrameIndex
returns: EntityIndex
---
1. Inventory person, object and event identifiers from identityRecords, canon and event participants.
   Collect exact statements equating names or declaring two things physically distinct.
2. Make AliasLinks for explicit equivalences. A pronoun can join an entity only within a compatible
   speaker/frame context with a unique antecedent; retain an unresolved note for competing antecedents.
3. Distinguish a real key from its rehearsal duplicate even when both are called the brass key.
   Similar appearance, common purpose and remembered possession do not establish object identity.
4. Build distinctPairs from explicit contrasts. Before joining alias classes, check the resulting
   class would not contain a distinct pair. Expose incompatible evidence instead of choosing a winner.
5. Normalize participant and object references to entity IDs while preserving all original names in
   AliasLink.names. Do not normalize a character's quoted nickname across speakers without support.
6. For event aliases, require explicit equivalence of action, frame and occurrence. Repeated actions
   by the same actor can be different events; an original event and proposed replacement are different.
7. Return aliases, distinctPairs and unresolved with source witnesses for every binding or exclusion.
   Empty evidence for identity leaves separate provisional entities rather than collapsing by spelling.
8. The finite alias relation can be closed over discovered links. Each join reduces separate classes;
   after all explicit links are consumed, stop and retain any unresolved referential question.
