/*---
engine: typescript-host
args:
  id: Text
  query: Text
returns: Text
---*/
return await host.research.nativeSearch(args.id, args.query);
