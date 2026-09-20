/*---
engine: typescript-host
args:
  run_id: Text
  index: Num
returns: TraceView
---*/
return host.ide.inspect(args.run_id, args.index);
