---
description: Proposes which slices and domains the next collection batch should cover, from the coverage of the admitted cases against their target shares. A proposal for a person; nothing is scheduled.
args:
  coverage: CoverageFacts
returns: CheckedBatchProposal
---
coverage lists, for the admitted cases, each slice and each domain with its target share, its count and its actual share, and the count of every slice and domain pair (cells). batch_size is the number of cases the next batch holds.

1. Read coverage.slices. For each slice compute target share minus actual share. Slices with a positive difference are under-covered; the largest difference is the most under-covered.
2. Do the same for coverage.domains.
3. Read coverage.cells. A pair whose count is 0, or far below its neighbours, is a gap.
4. Choose lines { slice, domain, count, rationale } for the batch: give each under-covered slice a count in proportion to its difference, and spread it over the domains that are under-covered, preferring the gaps of step 3. Use only the slice and domain names of coverage. A slice or domain whose target share is 0 gets no line.
5. The counts add up to at most batch_size. Each rationale is one sentence that names the shares or the gap that motivates the line.
6. Set note to one sentence on what remains uncovered after this batch.
Return { batch, note }.
