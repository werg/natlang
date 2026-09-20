/*---
engine: typescript-host
args:
  actor: Text
  round: Num
  plan: CombatPlan
returns: CombatReceipt
---*/
return host.combat.submit(args.actor, args.round, args.plan);
