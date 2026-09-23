import type { Answer, State } from "../../types.js";
export default function merge(state: State, found: string[][]): State {
const key = s => lower(s).replace(/[^a-z0-9 ]/g, "").trim()
const have = new Set(state.known.map(key))
const fresh = []
for (const f of found.flat()) if (!have.has(key(f))) { have.add(key(f)); fresh.push(f) }
return { known: state.known.concat(fresh), derived: state.derived.concat(fresh), grew: fresh.length > 0 }
}
