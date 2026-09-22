/*---
engine: typescript-host
args:
  actor: string
  round: number
  plan: CombatPlan
returns: CombatReceipt
---*/
return host.combat.submit(args.actor, args.round, args.plan);
