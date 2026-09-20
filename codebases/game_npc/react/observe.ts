/*---
engine: typescript-host
args:
  actor: Text
  event: NpcEvent
returns: NpcObservation
---*/
return host.npc.observe(args.actor, args.event);
