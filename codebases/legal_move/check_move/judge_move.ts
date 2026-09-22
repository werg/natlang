/*---
description: The verdict on placing the player's mark in a cell.
args:
  cells: Cell[]
  target: number
  player: Mark
returns: Verdict
---*/
const draw = c => [0, 3, 6].map(i => c.slice(i, i + 3).map(v => v === "empty" ? "." : v).join(" ")).join("\n")
const t = args.target
if (!Number.isInteger(t) || t < 1 || t > 9) return { legal: false, reason: `there is no cell ${t}`, wins: false, board: draw(args.cells) }
if (args.cells[t - 1] !== "empty") return { legal: false, reason: `cell ${t} is already taken by ${args.cells[t - 1]}`, wins: false, board: draw(args.cells) }
const next = args.cells.slice(); next[t - 1] = args.player
const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
const wins = lines.some(l => l.every(i => next[i] === args.player))
return { legal: true, reason: wins ? "legal, and it wins the game" : "legal", wins, board: draw(next) }
