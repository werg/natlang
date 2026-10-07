---
description: Filter. Keep the rows that meet an exact condition and every condition on meaning.
args:
  rows: Row[]
  where: string | null
  semantic: Semantic[]
returns: Row[]
---
Keep the rows that meet where, an SQL condition over their field names evaluated exactly in eval (as scan does).
Then apply each semantic condition. Judge it once for each distinct value its column has among the rows still kept:
meets(value as text, criterion), all at once. A row passes when every semantic condition's value meets its
criterion; a null value never does. Keep the rows' order.
