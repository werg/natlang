---
description: Build scoped world views from the same immutable intake.
args:
  intake: IntakeResult
returns: WorldIndex
---
1. Call parseFrames(intake.inventory). Keep its events attached to their declared frame IDs.
2. Call bindEntities(intake.inventory, frames). Resolve explicit aliases before joining temporal or
   object claims. Preserve stated distinctions between an original and its rehearsal duplicate.
3. Call orderEvents(frames, entities, intake.inventory.canon). The returned order is partial and
   frame-specific. File order is not event time unless a visible source convention declares it so.
4. Call evaluateAuthority(intake, frames, entities, temporal). Its facts and beliefsOnly remain
   separate arrays. Character reports do not become selected-world facts through repeated mention.
5. Check every event references an existing frame. A missing parent frame remains unresolved;
   do not attach that event to the selected world as a default.
6. Return {frames, entities, temporal, authority}. Each child result is consumed unchanged except
   for explicit coherence findings attached to its own unresolved collection.
7. World coherence precedes knowledge and custody. When order is unresolved, those later branches
   must represent alternative licensed possibilities rather than guessing an observation sequence.
8. The world branch may be reused on an edited private snapshot. It must reconstruct authority from
   that snapshot, so a candidate's self-description cannot dictate whether its new assertions are canon.
