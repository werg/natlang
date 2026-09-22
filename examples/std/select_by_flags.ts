/*---
description: The items whose flag (same position) is true.
args:
  items: string[]
  flags: boolean[]
returns: string[]
---*/
return args.items.filter((_, i) => args.flags[i])
