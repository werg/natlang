/*---
description: Say the line to the customer.
args:
  to: Text
  line: Text
  action: Action
returns: Bool
effects: [out.emit]
---*/
fx.out.emit({ to: args.to, line: args.line, action: args.action.code })
return true
