/*---
description: How many flags are true.
args:
  flags: Bool[]
returns: Num
---*/
return args.flags.filter(Boolean).length
