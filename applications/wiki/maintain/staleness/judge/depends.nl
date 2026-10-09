---
description: What a cell reads. Name the blocks, the other cells' results and the project files its result depends on.
args:
  cell: WikiBlock
  page: WikiPage
returns: Dependency
---
Work out what the result of cell, a cell block of page, depends on, from cell.text. page.blocks lists every block with
its ID and text.

1. blocks: the IDs of blocks of page whose text the cell is meant to work on. A cell names them with a [[block-id]]
   link or with words that point at one block (for example "the paragraph under Install"). A cell that works only on
   its own input has none.
2. cells: the IDs of other cell blocks of page whose result this cell uses, named the same way.
3. files: true when cell.language is "natlang" and its instructions use the project's files, false otherwise.

Return { cell: cell.id, blocks, cells, files }, with the IDs in the order of page.blocks.
