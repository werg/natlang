---
description: Select the notebook goal cell that can answer a user request.
args:
  request: string
  cells: Cell[]
returns: string
---
Choose exactly one offered cell ID whose result best answers the request. Use
descriptions and dependencies as evidence. Do not invent a cell. If several
cells contribute, select the final downstream result rather than an input.
