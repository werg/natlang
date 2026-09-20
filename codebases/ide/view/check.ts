/*---
engine: typescript-host
args:
  revision: Text
returns: CheckReport
---*/
return host.ide.check(args.revision);
