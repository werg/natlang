/*---
description: Why this position cannot be played from by this player; "" if it can.
args:
  cells: Cell[]
  player: Mark
returns: string
---*/
const c = args.cells
if (c.length !== 9) return `a board has 9 cells, this one has ${c.length}`
const x = c.filter(v => v === "X").length, o = c.filter(v => v === "O").length
if (x - o !== 0 && x - o !== 1) return `impossible position: ${x} X and ${o} O`
const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
const won = m => lines.some(l => l.every(i => c[i] === m))
if (won("X") || won("O")) return "the game is already over"
if (x + o === 9) return "the board is full"
const turn = x === o ? "X" : "O"
return turn === args.player ? "" : `it is ${turn}'s turn, not ${args.player}'s`
