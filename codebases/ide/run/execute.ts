/*---
engine: typescript-host
args:
  input: Text
  revision: Text
returns: RunReport
---*/
return await host.ide.run({ input: args.input }, args.revision);
