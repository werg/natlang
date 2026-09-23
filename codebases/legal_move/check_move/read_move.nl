---
description: Which cell (1-9, row by row) a move described in words refers to.
args:
  move: string
returns: number
---
`move` describes one tic-tac-toe move in words ("the centre", "bottom left", "top row, middle").
Answer the number of the cell, counting row by row from 1 (top-left) to 9 (bottom-right).
If it does not name exactly one cell, report a blocker saying what is unclear.
