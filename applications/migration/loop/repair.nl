---
description: Migration repair. Patches on top of a failing candidate that address the repairable findings - locate the place, classify it, write and check the patch, as for the first candidate.
args:
  intent: Intent
  snapshot: RepoSnapshot
  findings: Finding[]
uses: [site/locate, site/classify, site/edit]
returns: CheckedPatches
---
Repair the candidate at snapshot.revision from findings, each of them repairable. Work in eval, one finding at a time
in parallel with Promise.all.

For a finding:
1. Hit. When finding.path is not empty: hit = { path: finding.path, offset: 0, line: max(1, finding.line), excerpt: ""
   }. Otherwise hit = the first of repository.search(intent.old, snapshot.revision).hits; with no hit, the finding
   gives no patches.
2. site = await locate(intent, hit, snapshot.revision), with id "r" and the finding's position. usage = await
   classify(intent, site, finding.evidence), with action set to "edit": the finding is evidence that the site must
   change.
3. result = await edit(intent, { site, usage }, snapshot.revision, finding.evidence).
4. The finding's patches are result.patches (none when result.status is "failed").

Collect the patches of all findings, in the order of findings. When two patches have the same path and old text, keep
the first. Return them.
