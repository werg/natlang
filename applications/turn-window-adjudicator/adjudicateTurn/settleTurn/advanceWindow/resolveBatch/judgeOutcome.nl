---
description: Interpret one final effect part without observing any sibling unapplied outcome.
args:
  part: EventPart
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: PartOutcome
---
1. Read this actual rewritten part and its source-backed effect kind. Locate the named target instance
   and recipient in beforeBoard, preserving the distinction between a physical card and this garden stay.
   A missing target is resolved only by an explicit source rule; otherwise return an unresolved premise.
2. For damage, derive the target's proposed accumulated damage from its original before-state damage
   plus this nonnegative quantity. Record actualDamage from this final part. For growth, add growth
   counters only and record actualGrowth; integrity, damage and controller remain unchanged.
3. For ink, derive a resource proposal for the already bound recipient. A triggered recipient fixed
   when queued stays fixed even if a new instance or another controller now appears on the board.
4. For return, read the source-authorized new object carried by this part. Preserve the discarded old
   instance and create the distinct new instance with the stated initial properties. Reentry carries
   only explicitly granted fields; existing shared turn-use tokens retain their actual status.
5. For control or attachment, interpret exactly the instructed field change and preserve independent
   properties. Construct full proposed ObjectChanges with the same actual before objects; copy source
   witnesses supporting identity, effect meaning and selected scalar quantities.
6. Return PartOutcome with actual damage/growth, object/resource proposals, reason and unresolved.
   Batch reconciliation owns simultaneous destruction after all damage proposals have been combined;
   this per-part interpretation does not remove a sibling's before-state protection source.
