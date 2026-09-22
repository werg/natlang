/*---
engine: typescript-host
args:
  revision: string
returns: Validation
---*/
return await host.repository.validate(args.revision);
