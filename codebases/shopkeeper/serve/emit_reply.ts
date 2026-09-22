/*---
description: Say the line to the customer.
args:
  to: string
  line: string
  action: Action
returns: boolean
effects: [out.emit]
---*/
fx.out.emit({ to: args.to, line: args.line, action: args.action.code })
return true
