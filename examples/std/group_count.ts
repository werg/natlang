/*---
description: How often each value occurs.
args:
  values: Text[]
returns: Dict<Num>
---*/
const out = {}
for (const v of args.values) out[v] = (out[v] || 0) + 1
return out
