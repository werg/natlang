---
description: Run a pinned wiki cell in its declared child environment.
args:
  block_id: Text
  input: Text
returns: CellResult
---
function run_cell(block_id, input) -> CellResult
  return execute(block_id, input)
