/*---
description: The rules of a policy, one per non-empty line (list markers removed).
args:
  policy: Text
returns: Text[]
---*/
return args.policy.split("\n").map(l => l.replace(/^\s*(?:[-*]|\d+[.)])\s*/, "").trim()).filter(Boolean)
