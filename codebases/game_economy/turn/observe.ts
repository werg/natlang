/*---
engine: typescript-host
args:
  actor: string
returns: MarketObservation
---*/
return host.economy.observe(args.actor);
