/*---
description: Add newly found facts to the state; grew says whether anything was new.
args:
  state: State
  found: string[][]
returns: State
---*/
const key = s => lower(s).replace(/[^a-z0-9 ]/g, "").trim()
const have = new Set(args.state.known.map(key))
const fresh = []
for (const f of args.found.flat()) if (!have.has(key(f))) { have.add(key(f)); fresh.push(f) }
return { known: args.state.known.concat(fresh), derived: args.state.derived.concat(fresh), grew: fresh.length > 0 }
