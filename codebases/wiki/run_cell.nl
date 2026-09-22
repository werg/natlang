---
description: Run a pinned wiki cell in its declared child environment, with files available for linked artifacts.
args:
  block_id: Text
  input: Text
  files?: Dict<File>
returns: CellResult
---
function run_cell(block_id, input) -> CellResult
  return execute(block_id, input)
