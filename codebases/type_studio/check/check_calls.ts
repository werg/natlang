/*---
args:
  context: Context
  claims: CallClaim[]
returns: Diagnostic[]
effects: [types.calls]
---*/
return fx.types.calls(args.context, args.claims);
