/*---
engine: typescript-host
args:
  actor: string
  tick: number
  intent: TradeIntent
returns: TradeReceipt
---*/
return host.economy.submit(args.actor, args.tick, args.intent);
