---
description: Staleness of cell results. A result survives an edit when nothing it depends on changed, directly or through the cells it reads.
args:
  page: WikiPage
  delta: Delta
  records: CellRecord[]
returns: CellVerdict[]
---
Decide for each of records, the cell results the page remembers, whether the result is still fresh on page, the page
after an edit. delta lists the block IDs whose source changed, were added or were removed by the edit. The results
form a graph, and a verdict is derived from the verdicts of what the cell reads. Work in these steps.

1. For each record whose block is a cell of page, all at once: dependencies = depends(the cell block, page). A record
   whose block is no longer on page is stale with the reason "the cell was removed".
2. Order the cells so that a cell comes after every cell in its dependencies.cells. Cells that read each other in a
   cycle are all stale, with the reason "the cells read each other".
3. Take the cells in that order. A cell is stale when one of these holds, and the reason names it:
   - its own block ID is in delta.changed (its source changed);
   - a block ID in its dependencies.blocks is in delta.changed or delta.removed;
   - a cell in its dependencies.cells is stale.
   Otherwise it is fresh, with the reason "nothing it reads changed".
4. Return one CellVerdict { block_id, status, reason } for each record, in the order of records.
