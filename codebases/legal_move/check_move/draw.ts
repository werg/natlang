/*---
description: The board as three lines of text.
args:
  cells: Cell[]
returns: string
---*/
const s = args.cells.map(v => v === "empty" ? "." : v)
return [0, 3, 6].map(i => s.slice(i, i + 3).join(" ")).join("\n")
