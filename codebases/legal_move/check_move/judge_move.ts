export default function judge_move(cells: Cell[], target: number, player: Mark): Verdict {
const draw = c => [0, 3, 6].map(i => c.slice(i, i + 3).map(v => v === "empty" ? "." : v).join(" ")).join("\n")
const t = target
if (!Number.isInteger(t) || t < 1 || t > 9) return { legal: false, reason: `there is no cell ${t}`, wins: false, board: draw(cells) }
if (cells[t - 1] !== "empty") return { legal: false, reason: `cell ${t} is already taken by ${cells[t - 1]}`, wins: false, board: draw(cells) }
const next = cells.slice(); next[t - 1] = player
const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
const wins = lines.some(l => l.every(i => next[i] === player))
return { legal: true, reason: wins ? "legal, and it wins the game" : "legal", wins, board: draw(next) }
}
