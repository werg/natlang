/*---
engine: typescript-host
args:
  input: string
  revision: string
returns: RunReport
---*/
return await host.ide.run({ input: args.input }, args.revision);
