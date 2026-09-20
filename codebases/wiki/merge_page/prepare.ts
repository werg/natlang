/*---
engine: typescript-host
args:
  base: WikiPage
  updates: WikiUpdate[]
  profile: MergeProfile
returns: PreparedMerge
---*/
return host.wiki.prepare(args.base, args.updates, args.profile);
