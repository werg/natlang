/*---
engine: typescript-host
args:
  actor: string
  event: NpcEvent
returns: NpcObservation
---*/
return host.npc.observe(args.actor, args.event);
