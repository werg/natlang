/*---
engine: typescript-host
args:
  revision: string
returns: CheckReport
---*/
return host.ide.check(args.revision);
