/*---
engine: typescript-host
args:
  base: WikiPage
  prepared: PreparedMerge
returns: MergeReport
---*/
return { status: 'rejected', page: args.base, detail: args.prepared.detail };
