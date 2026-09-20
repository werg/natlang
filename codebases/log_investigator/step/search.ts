/*---
engine: typescript-host
args:
  observation: Observation
returns: Evidence[]
---*/
return host.logs.query(args.observation);
