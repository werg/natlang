/*---
description: How many flags are true.
args:
  flags: boolean[]
returns: number
---*/
return args.flags.filter(Boolean).length
