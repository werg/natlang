export default function group_count(values: string[]): Record<string, number> {
const out = {}
for (const v of values) out[v] = (out[v] || 0) + 1
return out
}
