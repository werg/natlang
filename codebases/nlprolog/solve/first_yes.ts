/*---
description: The first answer whose verdict is yes; unknown when there is none.
args:
  answers: Answer[]
returns: Answer
---*/
return args.answers.find(a => a.verdict === "yes") || { verdict: "unknown", proof: [] }
