/*---
engine: typescript-host
args:
  item: LogEvent
returns: Observation
---*/
return host.logs.observe(args.item);
