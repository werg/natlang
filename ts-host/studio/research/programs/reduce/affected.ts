/*---
engine: typescript-host
args:
  head: Text
  changed: Text[]
returns: Text[]
---*/
return await host.research.affected(args.head, args.changed);
