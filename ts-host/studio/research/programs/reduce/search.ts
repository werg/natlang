/*---
engine: typescript-host
args:
  head: Text
  query: Text
returns: Hit[]
---*/
return await host.research.search(args.head, args.query);
