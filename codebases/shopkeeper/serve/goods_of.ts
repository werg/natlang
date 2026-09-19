/*---
description: The names of the goods the shop sells.
args:
  acc: Shop
returns: Text[]
---*/
return Object.keys(args.acc.prices)
