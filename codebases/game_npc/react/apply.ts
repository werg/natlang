/*---
engine: typescript-host
args:
  observation: NpcObservation
  plan: NpcPlan
returns: NpcResult
---*/
return host.npc.apply(args.observation, args.plan);
