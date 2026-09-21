/*---
engine: typescript-host
args:
  id: Text
  offset: Num
  length: Num
returns: Text
---*/
return await host.research.nativeRead(args.id, args.offset, args.length);
