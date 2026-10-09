---
description: An NPC's policy for an event. Reply in character and choose one world action, from its memory, notes and commitments.
args:
  observation: NpcObservation
  notes: Note[]
returns: NpcPlan
---
Respond for the NPC observation.actor to observation.event. What it knows is in observation: its inventory, its memory
(the entries, the event among them as observation.event_id) and its commitments, and notes, what it concluded about
the event just now.

1. say is the reply, in the NPC's voice, built from the event, the memory, the notes and the commitments.
2. action is the one world action, chosen by these cases:
   - "give": the event asks for an item or the commitments call for it, and observation.inventory holds at least 1 of
     it. item is that item's name, target is the id of the recipient (usually event.from).
   - "promise": the event asks for help later. target is the recipient and detail is what the NPC will do.
   - "none": otherwise.
3. Return { say, action } with item and target for a give, or target and detail for a promise.
