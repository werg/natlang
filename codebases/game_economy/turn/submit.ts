/*---
engine: typescript-host
args:
  actor: Text
  tick: Num
  intent: TradeIntent
returns: TradeReceipt
---*/
return host.economy.submit(args.actor, args.tick, args.intent);
