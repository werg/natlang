/*---
engine: typescript-host
args:
  head: Text
returns: Entry[]
---*/
return await host.research.list(args.head);
