/*---
engine: typescript-host
args:
  actor: string
returns: CombatObservation
---*/
return host.combat.observe(args.actor);
