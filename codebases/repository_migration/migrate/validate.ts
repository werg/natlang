/*---
engine: typescript-host
args:
  revision: Text
returns: Validation
---*/
return await host.repository.validate(args.revision);
