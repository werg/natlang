---
description: Select the notebook goal cell that can answer a user request.
args:
  request: string
  cells: Cell[]
  finals: string[]
  problem?: string
returns: string
---
Choose the one cell whose result answers the request. finals lists the ids of cells that no other cell needs.

1. State in one phrase what the request asks for.
2. For each id in finals, read the description of that cell in cells and mark it a match when its result answers the request.
3. When exactly one id matches, return it. When several match, return the one whose description is closest to the wording of the request.
4. When no id in finals matches, repeat steps 2 and 3 over all ids in cells, using needs to prefer the cell that depends on the others.
5. Return an id that appears in cells, written exactly as it appears there. When problem is given, it states what was wrong with the previous answer; return an id that fixes it.
