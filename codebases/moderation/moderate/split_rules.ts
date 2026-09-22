/*---
description: The rules of a policy, one per non-empty line (list markers removed).
args:
  policy: string
returns: string[]
---*/
return args.policy.split("\n").map(l => l.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim()).filter(Boolean)
