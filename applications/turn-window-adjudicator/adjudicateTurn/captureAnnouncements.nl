---
description: Capture announcement-owned references and allocate the source-bounded work-slot universe.
args:
  request: TurnRequest
  source: TurnSource
  protocol: WindowProtocol
returns: CapturedTurn
---
1. Start from the actual returned initialBoard and submitted IntentReadings. Preserve this original
   revision as immutable; the listed responses are not applied until their later interrupt windows.
2. For each announced part, identify rules that will consume its target or controller reference.
   Call bindReference(part, consumerRule, initialBoard, source.clauses) for each such concrete consumer.
   Keep the returned bindings separate when consumers use captured, current or last-known control.
3. Build each Announcement from its original intent and parts plus these actual bindings. A captured
   controller is copied now from the original target; a live or last-known reference remains a typed
   deferred reference and must not receive the same copied value for convenience.
4. Allocate one WorkSlot for each actual submitted announcement and response. Preserve responseOrder
   from the source. An illustrative ability consequence is not allocated as a submitted player action.
5. Allocate one potential trigger WorkSlot for each source-defined one-use consequence token. Link it
   to the actual card/rule identity and initial token status. An already spent token starts consumed;
   an available trigger starts inactive until its condition is judged after resolution.
6. Form remainingWorkSlotIds from all unconsumed slots, including inactive future trigger slots.
   Form pendingSlotIds from currently active submitted work. The window protocol chooses which ready
   slot runs first; this inventory cannot bypass interrupts by executing the announced batch immediately.
7. Return initialBoard, announcements, responses, workSlots, pendingSlotIds, remainingWorkSlotIds,
   protocol and unresolved. Trigger emission later activates an already inventoried slot rather than
   increasing this finite universe. A source with unbounded new capacity is an explicit unsupported case.
