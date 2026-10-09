---
description: Update an NPC's memory for an event. Write the notes that tie the new event to what the NPC already remembers, each citing the entries it rests on.
args:
  observation: NpcObservation
returns: Note[]
---
Write the notes that the NPC observation.actor adds to its memory because of observation.event. The event is the
entry of observation.memory whose id is observation.event_id; the entries before it are what the NPC remembered.

1. Find the earlier entries that bear on the event: the same speaker (from), the same item or subject, or an earlier
   request, answer or promise the event follows up. Find the commitments that name the speaker.
2. For each connection worth keeping, write one note: text is one sentence in the NPC's own words saying what the NPC
   now takes the event to mean, and about lists the ids of the entries it rests on, always including
   observation.event_id.
3. Return the notes, at most three, in order of importance. An event that connects to nothing earlier gets one note
   that restates it, with about holding only the event's id.
