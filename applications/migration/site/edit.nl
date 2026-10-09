---
description: Migration edit of one site. Write the exact patch for a classified site, check it is exact, and write it again once with the problem when it is not.
args:
  intent: Intent
  classified: Classified
  revision: string
  problem?: string
returns: SiteEdit
---
Edit the site of classified at revision. patch writes the patches and exact checks them. Work in eval.

1. When classified.usage.action is "leave", return { site: classified.site.id, usage: classified.usage, patches: [],
   status: "left", note: classified.usage.reason }.
2. patches = await patch(intent, classified, revision, problem). verdict = await exact(patches, classified,
   revision).
3. When verdict.exact is false, patches = await patch(intent, classified, revision, verdict.problem), and verdict =
   await exact(patches, classified, revision) again.
4. Return { site: classified.site.id, usage: classified.usage, patches, status: "patched", note: classified.usage.reason
   } when verdict.exact. Otherwise return { site, usage, patches: [], status: "failed", note: verdict.problem }.
