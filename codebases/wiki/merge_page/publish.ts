/*---
engine: typescript-host
args:
  base: WikiPage
  prepared: PreparedMerge
  draft: MergeDraft
  profile: MergeProfile
returns: MergeReport
---*/
return host.wiki.publish(args.base, args.prepared, args.draft, args.profile);
