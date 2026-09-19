/*---
args:
  acc: State
  item: Event
returns: Bool
---*/
return args.acc.seen.includes(args.item.id);
