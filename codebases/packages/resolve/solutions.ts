/*---
engine: typescript-host
args:
  request: PackageRequest
returns: Lock[]
---*/
return host.packages.solutions(args.request);
