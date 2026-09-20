/*---
engine: typescript-host
args:
  request: Request
  plan: Plan
  receipt: Receipt
returns: Inspection
---*/
return await host.media.inspect(args.request, args.plan, args.receipt);
