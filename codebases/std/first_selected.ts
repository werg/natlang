/*---
description: The first item whose flag (same position) is true.
args:
  items: Text[]
  flags: Bool[]
returns: Text
---*/
return args.items[args.flags.indexOf(true)]
