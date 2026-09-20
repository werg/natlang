/*---
engine: typescript-host
args:
  revision: Text
  patch: Patch[]
returns: RepoSnapshot
---*/
return host.repository.apply(args.revision, args.patch);
