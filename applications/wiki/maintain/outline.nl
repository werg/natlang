---
description: Sections. Derive a page's outline from its heading blocks.
args:
  page: WikiPage
  scan: Scan
returns: Outline
---
Build the outline of page from scan.headings, the blocks that start with a heading. Walk page.blocks in order, with
these data structures: `sections` (the outline so far), `used` (the section IDs taken so far), `current` (the section
that the next block belongs to, none at first).

1. A block that has an entry in scan.headings starts a section: title and level from the heading. Its id is the slug
   of the title: lower case, every run of characters that are not letters or digits becomes one "-", no "-" at either
   end. When the slug is in `used`, append "-2", "-3" and so on until it is not. Its parent is the nearest earlier
   section with a smaller level, or null. Its block_ids start with this block's ID. It becomes `current`.
2. Any other block joins the block_ids of `current`. Blocks before the first heading form the section
   { id: "top", title: "", level: 0, block_ids, parent: null }, if there are any.
3. problems, one sentence each, naming the section:
   - two headings with the same slug before step 1 made them different;
   - a heading whose level is more than one deeper than its parent's;
   - a section whose block_ids hold only its own heading block.

Return { sections, problems }, with sections in page order.
