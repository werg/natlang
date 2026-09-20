---
description: Respond to an NPC event using assigned memory and legal world actions.
args:
  actor: Text
  event: NpcEvent
returns: NpcResult
---
function react(actor, event) -> NpcResult
  observation = observe(actor, event)
  plan = decide(observation)
  return apply(observation, plan)
