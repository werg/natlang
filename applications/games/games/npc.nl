---
description: An NPC reacts to one event. It updates its memory, plans a reply and one world action from that memory, and the village commits the event once.
args:
  state: NpcState
  actor: string
  event: NpcEvent
  settings: Settings
returns: Turn
---
Play the reaction of the NPC actor to event in the village state, with the stages in your folder. The reaction decides
on the snapshot state and then commits once; nothing changes before the commit.

1. observation = observe(state, actor, event). It holds the NPC's inventory, its memory with the event entered as
   observation.event_id, and its commitments.
2. notes = remember(observation, settings): the notes the NPC adds to its memory because of the event.
3. plan = respond(observation, notes).
4. verdict = validate(observation, plan, settings). When verdict.ok is false, replace plan by { say: plan.say, action:
   "none" } and add the log line "<actor>: <verdict.reason>".
5. effects = { basis: state.version, actor, event_id: observation.event_id, event, notes, plan }.
   committed = commit(state, effects).
6. When committed.ok is false, return { ok: false, state, events: [], log, problem: committed.problem }: the event is
   not acted on and state stays as it was.
7. Otherwise return { ok: true, state: committed.state, events: committed.events, log, problem: "" }.

log holds the line of step 4, if any, and, last, the line "<actor> <plan.action>" naming the action taken.
