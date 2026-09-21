/*---
engine: typescript-host
args:
  head: Text
  root: Text
  inputs: Text
  call_id: Text
returns: Receipt
---*/
return await host.research.execute(args.head, args.root, args.inputs, args.call_id);
