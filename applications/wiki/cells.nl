---
description: Cell evaluation policy. Decide which requested cells run now and in what order, and refuse the rest with a reason.
args:
  page: WikiPage
  requests: CellRequest[]
  fresh: string[]
returns: CellPlan
uses: [maintain/staleness/judge/depends]
---
Plan the evaluation of requests, the cells people or the wiki ask to run on page. fresh holds the IDs of cells whose
remembered result is still fresh. The plan is data; the wiki runs it. Work in these steps.

1. Refusals. Refuse a request, with the reason, when:
   - its block_id is not a cell of page ("no such cell");
   - its block_id is the block_id of a Conflict in page.unresolved ("its source is an unresolved conflict");
   - its origin is "auto" and the cell's language is "natlang" ("automatic runs use javascript cells only");
   - its origin is "auto" and its block_id is in fresh ("its result is fresh").
2. One run per cell. Among the requests that remain, a cell with several requests runs once: the first request whose
   origin is "user", or else the first request. A run is { block_id, input }.
3. Dependencies. For each run, all at once: depends(the cell block, page).
4. Batches. Put each run in a batch, by its cells among the runs: a run whose dependencies.cells holds no cell that
   is also being run goes in batch 0; any other run goes one batch after the latest batch of those cells. Take at most
   as many rounds as there are runs. Runs still unplaced after that read each other in a cycle: refuse them with the
   reason "the cells read each other". Within a batch, keep the order of page.blocks.
5. Return { batches, refused }.
