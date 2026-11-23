---
description: Turn a written rule into a reusable typed judgment function.
args:
  rule: string
  problems?: string[]
returns: "(item: unknown) => unknown"
---
Write a reusable natural-language function that applies rule to one input item and returns the answer rule specifies.
Keep the function's instructions self-contained, crisp, and faithful to rule. If problems is provided and non-empty,
they describe cases where an earlier version failed; adjust the instructions so those cases are judged accurately.
