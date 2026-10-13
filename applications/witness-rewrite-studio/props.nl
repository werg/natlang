---
description: Fold actual custody while preserving copies and framed appearances.
args:
  intake: IntakeResult
  world: WorldIndex
returns: PropIndex
---
1. Inventory physical objects and their initial holders from controlling canon. Resolve identity using
   world.entities, including its explicit distinctPairs for originals, rehearsal copies and namesakes.
2. Extract ObjectTransitions from selected events: transfer, appearance, copying or state change.
   Attach source and target holders only when the event actually establishes a custody change.
3. A remembered appearance creates a transition within that memory frame. It cannot update actual
   holders in the selected frame. A visual impression of a seal does not create another letter copy.
4. Order actual transitions by world.temporal. Fold supported custody changes into actualHolders
   at each event. Keep unresolved ownership where alternative orders or missing sources matter.
5. Check a transfer's source holder was supported beforehand. If not, retain the transition claim
   and an unresolved continuity defect; do not create an unseen handoff to repair it automatically.
6. Preserve object capability constraints from canon. A rehearsal key shaped like the original can
   appear in a scene without inheriting the original's ability to open the actual tide gate.
7. A character can know a message without owning its carrier. Do not infer a physical transfer
   from a later line demonstrating information access.
8. Return transitions, actualHolders and unresolved. These are physical contributions separate from
   actor beliefs; candidate audits must test both when an edit changes a line about possession.
9. The fold consumes the finite event inventory. Cycles or unresolved precedence yield explicit
   uncertainty, not another unbounded pass over the same events.
