/*---
engine: typescript-host
args:
  actor: Text
returns: MarketObservation
---*/
return host.economy.observe(args.actor);
