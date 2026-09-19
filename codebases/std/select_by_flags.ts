/*---
description: The items whose flag (same position) is true.
args:
  items: Text[]
  flags: Bool[]
returns: Text[]
---*/
return args.items.filter((_, i) => args.flags[i])
