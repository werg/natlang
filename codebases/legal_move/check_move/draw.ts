export default function draw(cells: Cell[]): string {
const s = cells.map(v => v === "empty" ? "." : v)
return [0, 3, 6].map(i => s.slice(i, i + 3).join(" ")).join("\n")
}
