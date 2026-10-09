---
description: Page structure maintenance. After an edit, derive the page's sections, resolve its links, propose repairs for renamed targets and judge which cell results are still fresh.
args:
  before: WikiPage
  after: WikiPage
  records: CellRecord[]
  settings: WikiSettings
returns: Maintenance
---
Maintain the structure of `after`, a wiki page, which `before` was the revision before. records are the cell results the
page remembers. Every derived value is computed from the snapshots `before` and `after`, in this order, and each
step uses only the values of the steps above it. Use the stages in your folder.

1. Scans. scan(after) and scan(before), both at once.
2. Outlines. outline(after, the scan of after) and outline(before, the scan of before), both at once.
3. Links. For each link of the scan of after, all at once: resolve(link, the outline of after, the outline of before,
   the IDs of after's blocks).
4. Repairs. For each block with links of status "renamed": text = the block's text, with retarget(text, target,
   resolves_to) applied for each of its renamed links. Each repaired block gives one update
   { id: "repair-" + block ID, block_id, base_revision: after.revision, author: "wiki", text }.
5. Cell results. verdicts = staleness(after, delta(before, after), records, settings).
6. Return { structure: { revision: after.revision, outline: the outline of after, links: all the resolved links,
   cells: verdicts }, repairs }.
