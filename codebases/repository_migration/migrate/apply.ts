/*---
engine: typescript-host
args:
  revision: string
  patch: Patch[]
returns: RepoSnapshot
---*/
return host.repository.apply(args.revision, args.patch);
