/*---
args:
  acc: State
  item: Event
returns: boolean
---*/
return args.acc.seen.includes(args.item.id);
