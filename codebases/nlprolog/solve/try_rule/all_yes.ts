/*---
description: yes (with the rule and all sub-proofs as proof) if every result is yes, else unknown.
args:
  results: Answer[]
  rule: Text
returns: Answer
---*/
const ok = args.results.length > 0 && args.results.every(r => r.verdict === "yes")
return ok ? { verdict: "yes", proof: [args.rule].concat(...args.results.map(r => r.proof)) } : { verdict: "unknown", proof: [] }
