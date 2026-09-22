/*---
engine: typescript-host
args:
  run_id: string
  index: number
returns: TraceView
---*/
return host.ide.inspect(args.run_id, args.index);
