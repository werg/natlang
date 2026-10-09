---
description: Migration planning, survey. Every site in the repository that mentions the old thing - search evidence merged and widened into whole statements, with exact text.
args:
  intent: Intent
  revision: string
uses: [site/locate]
returns: CheckedSite[]
---
Find the sites of intent at revision. The repository service searches and reads exactly; locate widens one hit. Work
in eval.

1. Search. For every query in intent.queries, results = repository.search(query, revision). hits = all their hits,
   without duplicates (the same path and offset once).
2. Group. Sort hits by path, then line. Hits in the same file on the same line form one group. Take the first hit of
   each group.
3. Locate. sites = await Promise.all of locate(intent, hit, revision) for the first hit of each group.
4. Merge. Sort sites by path, then from. When a site starts at or before the end (to) of the previous site in the same
   file, merge them: from is the smaller from, to the larger to, text = repository.lines(path, from, to, revision),
   hits the sum.
5. Number. The ids are "s1", "s2", … in that order. Return the sites.

When there are no hits, return [].
